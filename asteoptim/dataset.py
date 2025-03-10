from xmitgcm import open_mdsdataset
from xmitgcm.utils import get_extra_metadata
import numpy as np
import xarray as xr
import glob
import functools

# custom routine to properly load aste1080, which is formatted differently from other aste models
original_get_extra_metadata = get_extra_metadata  

def override_get_extra_metadata(func):
    @functools.wraps(func)
    def wrapper(domain='aste', nx=None):
        if nx == 1080:
            return get_extra_metadata_aste1080()
        return func(domain, nx)
    return wrapper

def get_extra_metadata_aste1080():
    em = get_extra_metadata(domain='aste', nx=270)
    em['ny'] = 4140
    em['nx'] = 1080
    em['ny_facets'] = [1260, 0, 1080, 540, 1260]
    em['pad_before_y'] = [int(1080/3)+270+270, 0, 0, 0, 0]
    em['pad_after_y'] = [0, 0, 0, int(1080/3)+270, 1080]
    return em

# Apply override
get_extra_metadata = override_get_extra_metadata(get_extra_metadata)

default_openmdsdataset_kwargs = {
            'iters': None,
            'read_grid': True,
            'geometry': 'llc',
            'default_dtype': np.float32
        }

def open_astedataset(data_dir, nx=270, face_to_tile=True, **open_mdsdataset_kwargs):
    """
    Wrapper function to open an xmitgc aste domain dataset.
    
    Parameters:
    - data_dir: Directory containing the dataset
    - nx: The x dimension size (default is 270)
    - face_to_tile: Boolean indicating whether to rename the 'face' dimension to 'tile' (default is True)
    - open_mdsdataset_kwargs: Additional arguments passed to the open_mdsdataset function, allowing customization.
    
    Returns:
    - A dataset (xarray Dataset) with the requested transformations and renaming.
    """
    
    # Merge default kwargs with the user-supplied kwargs (if any)
    open_mdsdataset_kwargs = {**default_openmdsdataset_kwargs, **open_mdsdataset_kwargs}
    
    # Add extra metadata specific to 'aste' domain and nx size
    open_mdsdataset_kwargs['extra_metadata'] = get_extra_metadata(domain='aste', nx=nx)
    open_mdsdataset_kwargs['nx'] = nx  # Include the nx value in the kwargs
    
    if 'prefix' in open_mdsdataset_kwargs.keys() and open_mdsdataset_kwargs['iters'] is None:
        open_mdsdataset_kwargs.pop('iters')
    
    ds = open_mdsdataset(data_dir, **open_mdsdataset_kwargs)
    
    if face_to_tile:
        ds = ds.rename({'face': 'tile'})
    
    return ds


def open_asteoptimdataset(run_dir_root,
                     optim_iters=None,
                     opt_subdir_leading_zeros=True, **open_mdsdataset_kwargs):
    """
    open_asteoptimdataset - Load xarray datasets from multiple optimization subdirectories of a root run directory and concatenate them along a new dimension 'ioptim'.
    
    Parameters:
        run_dir_root (str): Root directory for the run with subdirectories iterXXXX
        optim_iters (list, optional): Indices of iterations to load. Defaults to the total number of iterations in the run directory.
        **open_mdsdataset_kwargs: Additional keyword arguments to be passed to open_astedataset().
    
    Returns:
        xr.Dataset: Concatenated xarray dataset along the 'ioptim' dimension.
    
    Example:
        ds = load_ds_opt("/path/to/run/", nopts=5, prefix="state_3d_set", delta_t=600)
    """

    opt_subdir = 'iter{:04d}/diags/{}/' if opt_subdir_leading_zeros else 'iter{}/diags/{}/'

    if optim_iters is None:
        nopts = len(glob.glob(run_dir_root + 'iter*'))
        optim_iters = range(nopts)

    prefix = open_mdsdataset_kwargs['prefix'][0]

    ds_list = []


    for ioptim in optim_iters:
        data_dir = run_dir_root + opt_subdir.format(ioptim, prefix)
        open_mdsdataset_kwargs['data_dir'] = data_dir
        ds_ioptim = open_astedataset(**open_mdsdataset_kwargs)
        ds_list.append(ds_ioptim)

    ds=xr.concat(ds_list, dim="ioptim")
    return ds
