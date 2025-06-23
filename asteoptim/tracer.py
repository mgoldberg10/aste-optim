import xarray as xr
import numpy as np
import dask.array as da
from gcmfacespy import convert2widefaces
from ecco_v4_py.llc_array_conversion import llc_tiles_to_faces, llc_faces_to_tiles


@xr.register_dataarray_accessor('at')
class AsteTracer:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self):
        return get_aste_tracer_xr(self._obj)

@xr.register_dataarray_accessor('to_compact')
class AsteCompact:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self):
        return get_aste_xr_to_compact(self._obj)


def get_aste_xr_to_compact(data_array):
    data_array_tracer = data_array.at().values
    nx = int(data_array_tracer.shape[1]/2)
    nfx = [nx, 0, nx, int(2*nx/3), int(5*nx/3)]
    nfy = [int(5*nx/3), 0, nx, nx, nx]
    data_array_compact = aste_tracer2compact(data_array_tracer, nfx, nfy).squeeze()
    data_array_compact = xr.DataArray(data_array_compact, dims=['j', 'i'])
    return data_array_compact
    


def get_aste_tracer_xr(data_array):
    """
    Transform an xarray.DataArray of size (ntile, nx, nx) or (extra_dims, ntile, nx, nx)
    to (extra_dims, 900, 540), while keeping the operation lazy.

    Parameters:
    data_array : xarray.DataArray
        Input data array of shape (ntile, nx, nx) or (extra_dims, ntile, nx, nx).

    Returns:
    xarray.DataArray
        Transformed data array of shape (extra_dims, 900, 540) with appropriate rotations
        and rearrangements applied to the input tiles.
    """
    dims = data_array.dims
    j_dim = next(dim for dim in dims if dim in ['j', 'j_g'])
    i_dim = next(dim for dim in dims if dim in ['i', 'i_g'])
    extra_dims = [dim for dim in dims if dim not in {j_dim, i_dim, 'tile'}]
    nx = data_array.sizes[i_dim]

    # Initialize lazily filled arrays for each transformed face
    f0 = data_array.isel({j_dim: slice(int(nx / 3), None), 'tile': 0})
    f0 = assign_coords_to_dataarray(f0, i_dim, j_dim)

    f5 = data_array.isel({'tile': 5}).transpose(*extra_dims, i_dim, j_dim).isel({i_dim: slice(None, None, -1)}).isel({i_dim: slice(int(nx / 3), None)})
    f5 = assign_coords_to_dataarray(f5, i_dim, j_dim)

    # Concatenate f5 and f0 along 'I' axis
    f50 = xr.concat([f5.rename({i_dim: 'J', j_dim: 'I'}), f0.rename({j_dim: 'J', i_dim: 'I'})], dim='I')
    f50 = f50.reset_index('I', drop=True)

    # Process f1 and f4
    f1 = data_array.isel({'tile': 1})
    f4 = data_array.isel({'tile': 4}).transpose(*extra_dims, i_dim, j_dim).isel({i_dim: slice(None, None, -1)})
    f1 = assign_coords_to_dataarray(f1, i_dim, j_dim)
    f4 = assign_coords_to_dataarray(f4, i_dim, j_dim)

    # Concatenate f4 and f1 along 'I' axis
    f41 = xr.concat([f4.rename({i_dim: 'J', j_dim: 'I'}), f1.rename({j_dim: 'J', i_dim: 'I'})], dim='I')
    f41 = f41.reset_index('I', drop=True)

    # Process f3 and f2
    f3 = data_array.isel({'tile': 3})
    f3.data = da.rot90(f3.data, k=1, axes=(-2, -1))
    f3 = f3.transpose(*extra_dims, i_dim, j_dim).isel({j_dim: slice(None, None, -1)})
    f2 = data_array.isel({'tile': 2}).transpose(*extra_dims, i_dim, j_dim).isel({j_dim: slice(None, None, -1)})
    f2 = assign_coords_to_dataarray(f2, i_dim, j_dim)
    f3 = assign_coords_to_dataarray(f3, i_dim, j_dim)

    # Concatenate f2 and f3 along 'J' axis
    f23 = xr.concat([f2.rename({i_dim: 'J', j_dim: 'I'}), f3.rename({j_dim: 'J', i_dim: 'I'})], dim='J')
    f23 = f23.reset_index('J', drop=True)
    f23 = f23.isel({'J': slice(0, int(nx * 5 / 3))})

    # Create an empty DataArray with NaN values, same shape as f23
    empty = xr.DataArray(
        np.full_like(f23, fill_value=np.nan),
        dims=f23.dims,
        coords=f23.coords
    )


    # Concatenate the empty array and f23 along 'I' axis
    fe23 = xr.concat([empty, f23], dim='I')
    fe23 = fe23.reset_index('I', drop=True)
    fe23 = fe23.assign_coords(J=fe23['J'])

    ## Final concatenation along 'J' axis
    tracer_das = [f50, f41, fe23]
    aste_tracer = xr.concat(tracer_das, dim='J')
    aste_tracer = aste_tracer.reset_index('J', drop=True)

    return aste_tracer
    
def assign_coords_to_dataarray(data_array, i_dim, j_dim):
    """
    Assign coordinates to a data array for i_dim and j_dim.

    Parameters:
    data_array : xarray.DataArray
        Input data array that needs coordinates assigned.
    i_dim : str
        The name of the i-coordinate dimension.
    j_dim : str
        The name of the j-coordinate dimension.

    Returns:
    xarray.DataArray
        The input data array with coordinates assigned to i_dim and j_dim.
    """
    return data_array.assign_coords({
        i_dim: np.arange(len(data_array[i_dim])),
        j_dim: np.arange(len(data_array[j_dim])),
    })


def aste_global2tracer(array_global, nx=270):
    pad = nx // 3
    # Determine output shape, which matches the input shape for all but the last two dimensions
    output_shape = array_global.shape[:-2] + (10 * pad, 6 * pad)

    # Initialize the output array with zeros, same type as input
    array_tracer = np.zeros(output_shape, dtype=array_global.dtype)

    # Assign the corresponding slices
    array_tracer[..., :5 * pad, nx:] = array_global[..., nx + pad:3 * nx, :nx]
    array_tracer[..., :5 * pad, :nx] = array_global[..., nx + pad:3 * nx, 3 * nx:]
    array_tracer[..., 5 * pad:5 * pad + nx, nx:] = array_global[..., 3 * nx:, :nx]

    # Set NaNs in the bottom left portion
    array_tracer[..., 5 * pad:, :nx] = np.nan

    # Flip and assign the bottom right portion
    array_tracer[..., 8 * pad:, nx:] = np.flip(array_global[..., 2 * nx + pad:3 * nx, 2 * nx:3 * nx], axis=-1)

    return array_tracer


def get_aste_tracer_xr_inv(dask_array):
    """
    Transform an xarray.DataArray of size (extra_dims, 900, 540) to (extra_dims, ntile, nx, nx).
    
    Parameters:
    da : xarray.DataArray
        Input data array of shape (extra_dims, 900, 540).
    
    Returns:
    xarray.DataArray
        Transformed data array of shape (extra_dims, ntile, nx, nx) with appropriate rotations
        and rearrangements applied to the input tiles.
    """

    # Identify the horizontal dimension names
    dims = dask_array.dims
    J_dim, I_dim = ['J', 'I']
    ntile = 6
    nx = int(len(dask_array[I_dim]) / 2)
    extra_dims = [dim for dim in dims if dim not in {J_dim, I_dim}]
    
    # Initialize an empty DataArray to hold the transformed data
    original_tracer = xr.DataArray(
        0.,
        dims=extra_dims + ["tile", "j", "i"],
        coords={**{dim: dask_array.coords[dim] for dim in extra_dims}, "tile": np.arange(ntile), "j": np.arange(nx), "i": np.arange(nx)}
    )
    f0 = dask_array.loc[{J_dim: slice(0, int(nx * 2 / 3)), I_dim: slice(nx, None)}]
    f1 = dask_array.loc[{J_dim: slice(int(nx * 2 / 3), int(nx * 2 / 3)+nx), I_dim: slice(nx, None)}]
    f2 = dask_array.loc[{J_dim: slice(int(nx * 2 / 3)+nx, int(nx * 2 / 3)+2*nx), I_dim: slice(nx, None)}].transpose(*extra_dims, I_dim, J_dim).isel({I_dim: slice(None, None, -1)})
    f3 = dask_array.loc[{J_dim: slice(int(nx * 8 / 3), None), I_dim: slice(nx, None)}].\
    transpose(*extra_dims, I_dim, J_dim).\
    isel({I_dim: slice(None, None, -1)})
    f4 = dask_array.loc[{J_dim: slice(int(nx * 2 / 3), int(nx * 2 / 3)+nx), I_dim: slice(0, nx)}].transpose(*extra_dims, I_dim, J_dim).isel({J_dim: slice(None, None, -1)})
    f5 = dask_array.loc[{J_dim: slice(0, int(nx * 2 / 3)), I_dim: slice(0, nx)}].\
    transpose(*extra_dims, I_dim, J_dim).\
    isel({J_dim: slice(None, None, -1)})
    
    original_tracer.loc[dict({dim: slice(None) for dim in extra_dims}, **{'tile': 0, 'j': slice(int(nx / 3), None)})] = f0.rename({J_dim: 'j', I_dim: 'i'})
    original_tracer.loc[dict({dim: slice(None) for dim in extra_dims}, **{'tile': 1})] = f1.rename({J_dim: 'j', I_dim: 'i'})
    original_tracer.loc[dict({dim: slice(None) for dim in extra_dims}, **{'tile': 2})] = f2.rename({I_dim: 'j', J_dim: 'i'})
    original_tracer.loc[dict({dim: slice(None) for dim in extra_dims}, **{'tile': 3, 'i': slice(0, int(nx * 2 / 3)-1)})] = f3.rename({I_dim: 'j', J_dim: 'i'})
    original_tracer.loc[dict({dim: slice(None) for dim in extra_dims}, **{'tile': 4})] = f4.rename({I_dim: 'j', J_dim: 'i'})
    original_tracer.loc[dict({dim: slice(None) for dim in extra_dims}, **{'tile': 5, 'i':slice(0, int(nx * 2 / 3)-1)})] = f5.rename({I_dim: 'j', J_dim: 'i'})

    return original_tracer

def aste_faces2compact(fld,nfx=[270, 0, 270, 180, 450],nfy=[450, 0, 270, 270, 270]):
    '''
    Reverse of get_aste_faces, taking an input field from tracer form to compact form
    '''

    #add a new dimension in case it's only 2d field:
    sz=np.shape(fld.f1)
    sz=np.array(sz)
    if(len(sz)<3):
       sz=np.append(sz,1)

    nz=sz[0]
    nx=sz[-1]

    fldo = np.full((nz, 2 * nfy[0] + nx + nfx[3], nx), np.nan)

    if nz == 1:
        fld.f1=fld.f1[np.newaxis, :, :]
    fldo[:,0:nfy[0],:]=fld.f1
    if nz == 1:
        fld.f3=fld.f3[np.newaxis, :, :]
    fldo[:,nfy[0]:nfy[0]+nfy[2],:]=fld.f3
    if nz == 1:
        fld.f4=fld.f4[np.newaxis, :, :]
    fldo[:,nfy[0]+nfy[2]:nfy[0]+nfy[2]+nfx[3],:]=np.reshape(fld.f4,[nz,nfx[3],nfy[3]])
    if nz == 1:
        fld.f5=fld.f5[np.newaxis, :, :]
    fldo[:,nfy[0]+nfy[2]+nfx[3]:nfy[0]+nfy[2]+nfx[3]+nfx[4],:]=np.reshape(fld.f5,[nz,nfx[4],nfy[4]])

    return fldo

class structtype:
    pass

def aste_tracer2compact(fld, nfx=[270, 0, 270, 180, 450], nfy=[450, 0, 270, 270, 270]):
    '''
    Reverse of get_aste_tracer function
    Inputs:
        fld: the field in tracer form [nx*2 nfy(1)+nfy(3)+nfx(4)+nfx(5),nz]
        nfx: number of x faces
        nfy: number of y faces

    Outputs:
        fldout: the original data field in compact form, useful for comparison with read binary files
        Out: compact format [nz 1350 270]
    '''
    # check and fix if 2D
    sz=np.shape(fld)
    sz=np.array(sz)

    #add a new dimension in case it's only 2d field:
    if(len(sz)<3):
        sz=np.append(1,sz)
        fld=fld[np.newaxis, :, :]
    
    nz=sz[0]
    nx=sz[-1]
    
    nx = nfx[2]
    tmp1 = fld[:,:nfy[0],nx:]

    # cw rotation
    tmp3 = fld[:,nfy[0]:nfy[0]+nx,nx:]
    tmp3=np.transpose(tmp3, (1,2,0))
    tmp3 = list(zip(*tmp3))[::-1]
    tmp3 = np.asarray(tmp3)
    tmp3 = np.transpose(tmp3,[2,0,1])

    # cw rotation
    tmp4 = fld[:,nfy[0]+nx:,nx:]
    tmp4=np.transpose(tmp4, (1,2,0))
    tmp4 = list(zip(*tmp4))[::-1]
    tmp4 = np.asarray(tmp4)
    tmp4 = np.transpose(tmp4,[2,0,1])

    # ccw rotation
    tmp5 = fld[:,0:nfy[0],0:nx]
    tmp5=np.transpose(tmp5, (1,2,0))
    tmp5 = list(zip(*tmp5[::-1]))
    tmp5 = np.asarray(tmp5)
    tmp5 = np.transpose(tmp5,[2,0,1])

    tmp_struct = structtype()
    tmp_struct.f1 = tmp1
    tmp_struct.f3 = tmp3
    tmp_struct.f4 = tmp4
    tmp_struct.f5 = tmp5

    compact = aste_faces2compact(tmp_struct,nfx,nfy)

    return compact


def aste_compact_to_global270(v0, mygrid, less_output=True, return_faces=True):
    # taken from gcmfaces

    nFaces = len(mygrid.facesSize)
    if v0.ndim == 2:
        v0 = v0[np.newaxis, :, :]
    n3, n2, n1, n4, n5 = v0.shape + (1,) * (nFaces - len(list(v0.shape)))
    v00 = v0.reshape(n3 * n4 * n5, n1 * n2)
    i0 = 0
    i1 = 0
    v1 = dict()
    c = 0

    for iFace in range(nFaces):
        nn, mm = mygrid.facesSize[iFace, :]
        i0 = i1
        i1 = i1 + nn * mm
        this_v1 = v00[:,i0:i1].reshape(n3, mm, nn, n4, n5)
        this_v1 = this_v1.transpose(2, 1, 0, 3, 4)
        v1[iFace+1] = this_v1
    v1 = convert2widefaces(v1, mygrid)

    for iFace in range(nFaces):
        v1[iFace+1] = np.squeeze(v1[iFace+1])
        tpose_shape = (v1[iFace+1].ndim == 3) * (2,) + (1, 0)
        v1[iFace+1] = v1[iFace+1].transpose(tpose_shape)

    array_tiles = llc_faces_to_tiles(v1, less_output=less_output)

    if return_faces:
        return v1
    else:
        return array_tiles

def get_aste_faces(fld,nfx=[270, 0, 270, 180, 450],nfy=[450, 0, 270, 270, 270]):
    '''
    From big ASTE, get the data on the individual faces from the ASTE grid in case we want to observe individually
    input fld (of shape from rdmds reshaped to ny,nx or nz,ny,nx)
    '''
    nx=nfx[0]
    
    #check the klevel dimension, if 2d, add a third dim
    sz=np.shape(fld)
    sz=np.array(sz)
    if(len(sz)<3):
        fld=np.copy(fld[np.newaxis,:,:])
    
    tmp = fld[:,0:nfy[0],0:nx]
    fldout = dict()
    fldout[1]=fld[:,0:nfy[0],0:nx]                    #face 1
    fldout[3]=fld[:,nfy[0]:nfy[0]+nfy[2],0:nx]        ##face 3
    fldout[2]=np.array([])
    fldout[4]=np.reshape(fld[:,nfy[0]+nfy[2]:nfy[0]+nfy[2]+nfx[3],0:nx],[-1,nx,nfx[3]]) ##face 4
    fldout[5]=np.reshape(fld[:,nfy[0]+nfy[2]+nfx[3]:nfy[0]+nfy[2]+nfx[3]+nfx[4],0:nx],[-1,nx,nfx[4]]) ##face 5
    return fldout

def plot_aste_faces(fld,klev,climit,step, nfx=[270, 0, 270, 180, 450],nfy=[450, 0, 270, 270, 270]):
    '''
    Plots faces 1-4 of the ASTE grid, 
    input
        fld: must be from rdmds do not edit or reshape this
    '''
    fldout=get_aste_faces(fld,nfx,nfy=[450, 0, 270, 270, 270])
    nx=nfx[0]
    #step=(climit[1]-climit[0])/100
    print(step)
    clevels = np.arange(climit[0], climit[1], step)
    fig,axs=plt.subplots(2,2)
    pcm=axs[0,0].contourf(fldout.f1[klev-1,:,:],levels=clevels, cmap='viridis')
    fig.colorbar(pcm,ax=axs[0,0],location='right')
    axs[0,0].title.set_text('fld face1')
    pcm=axs[0,1].contourf(fldout.f3[klev-1,:,:],levels=clevels,cmap='viridis')
    fig.colorbar(pcm,ax=axs[0,1],location='right')
    axs[0,1].title.set_text('fld face3')
    pcm=axs[1,0].contourf(fldout.f4[klev-1,:,:],levels=clevels,cmap='viridis')
    fig.colorbar(pcm,ax=axs[1,0],location='right')

#def get_aste_tracer(fldin,nfx=[270, 0, 270, 180, 450],nfy=[450, 0, 270, 270, 270]):
def get_aste_tracer(fldin,nx=270):
    '''
    Inputs:
        fldin: data field in compact coords from mitgcm output (of shape from rdmds reshaped to ny,nx or nz,ny,nx)
        nx: number of points in model llc grid spanning a quarter circumference at the equator

    outputs:
        the input field reshaped into tracer form, plottable in xyz space

    '''
    
#    nfx: number of x faces, nfx = np.array([nx, 0 , nx, ncut2 ,ncut1])
#    nfy: number of y faces, nfy = np.array([ncut1, 0 , nx, nx, nx])
    nfx = [nx, 0, nx, int(2*nx/3), int(5*nx/3)]
    nfy = [int(5*nx/3), 0, nx, nx, nx]
    
    sz=np.shape(fldin)
    sz=np.array(sz)
    if(len(sz)<3):
       sz=np.append(1,sz)
    
    nz=sz[0]
    nx=sz[-1]

    
    #add a new dimension in case it's only 2d field:
    if nz == 1:
        fldin=fldin[np.newaxis, :, :]
    #defining a big face:
    a = np.full((nz, nfy[0]+nx+nfx[3], 2*nx), np.nan)
    
    #face1
    tmp=fldin[:,0:nfy[0],0:nx]        #(50,450,270)
    a[:,0:nfy[0],nx:2*nx]=tmp
    # return a
    
    #face3
    tmp=fldin[:,nfy[0]:nfy[0]+nx,0:nx] #(50, 270,270)
    tmp=np.transpose(tmp, (1,2,0))     #(270,270,50)
    ##syntax to rotate cw:
    tmp1=list(zip(*tmp[::-1]))         #type is <class 'zip'> --> <class 'list'>
    tmp1=np.transpose(tmp1,[2,0,1])    #(50,270,270)
    a[:,nfy[0]:nfy[0]+nx,nx:2*nx]=tmp1
    
    #face4
    tmp=np.reshape(fldin[:,nfy[0]+nx:nfy[0]+nx+nfx[3],0:nx],[nz,nx,nfx[3]]) #(50,270,180)
    tmp=np.transpose(tmp, (1,2,0))
    #syntax to rotate cw:
    tmp1=list(zip(*tmp[::-1]))      #type is <class 'list'>
    tmp1=np.asarray(tmp1)           #type <class 'numpy.ndarray'>, shape (180,270,50)
    tmp1=np.transpose(tmp1,[2,0,1]) #(50,180,270)
    a[:,nfy[0]+nx:nfy[0]+nx+nfx[3],nx:2*nx]=tmp1
    
    #face5
    tmp=np.reshape(fldin[:,nfy[0]+nx+nfx[3]:nfy[0]+nx+nfx[3]+nfx[4],0:nx],[nz,nx,nfx[4]]) #(50,270,450)
    tmp=np.transpose(tmp, (1,2,0))
    #syntax to rotate ccw:
    tmp1=list(zip(*tmp))[::-1]      #type is <class 'zip'> --> <class 'list'>
    tmp1=np.asarray(tmp1)           #type <class 'numpy.ndarray'>, shape (450,270,50)
    tmp1=np.transpose(tmp1,[2,0,1]) #(50,450,270)
    a[:,0:nfx[4],0:nx]=tmp1
    
    return a

