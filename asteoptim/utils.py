import xarray as xr
import numpy as np
import dask.array as da

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
    f4 = assign_coords_to_dataarray(f4, i_dim, j_dim)

    # Concatenate f4 and f1 along 'I' axis
    f41 = xr.concat([f4.rename({i_dim: 'J', j_dim: 'I'}), f1.rename({j_dim: 'J', i_dim: 'I'})], dim='I')
    f41 = f41.reset_index('I', drop=True)

    # Process f3 and f2
    f3 = data_array.isel({'tile': 3})
    f3.data = da.rot90(f3.data, 1)
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
        np.full_like(f23, fill_value=np.nan),  # Create an array of NaN with the same shape
        dims=f23.dims,                        # Use the same dimensions as f2
        coords=f23.coords                     # Use the same coordinates as f2
    )

    # Concatenate the empty array and f23 along 'I' axis
    fe23 = xr.concat([empty, f23], dim='I')
    fe23 = fe23.reset_index('I', drop=True)

    # Final concatenation along 'J' axis
    aste_tracer = xr.concat([f50, f41, fe23], dim='J')
    aste_tracer = aste_tracer.reset_index('J', drop=True)

    return aste_tracer

def get_fH(ds, g=9.81, tau=86164):
    # coriolis
    H = ds.Depth
    lat = ds.YC
    Omega = (2 * np.pi) / tau
    lat_rad = (np.pi / 180) * lat  # convert latitude from degrees to radians
    f = 2 * Omega * np.sin(lat_rad)
    return f, H
