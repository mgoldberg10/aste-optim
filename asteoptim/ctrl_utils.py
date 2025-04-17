from smartcables import *
from .dataset import open_astedataset
import xarray as xr
from xmitgcm.variables import package_state_variables
from ecco_v4_py import get_llc_grid
import matplotlib.pyplot as plt
from ecco_v4_py.vector_calc import *


def load_gentim2d_ds(run_dir, ctrl_vars, optim_iters=range(1, 2), ext='', thresh=1e5):
    # note: thresh is by no means a foolproof way to do this -- can't think of a better way to determine which xx_gentim2d records correspond to a given run
    # The other way is that the user could supply modelStart and modelEnd dates and gentim2d periods
    
    ds = xr.Dataset()
    for ic, ctrl_var in enumerate(ctrl_vars):
        # load adxx(iternum=0) just to get recs
        iternum = 0
        xx_var = f'adxx_{ctrl_var}'
        xx_fname = f'{run_dir}/iter{iternum:04d}/{xx_var}'
        meta = get_aste_file_metadata(xx_fname, iternum=iternum, dtype=np.dtype('>f4'))
        xx = read_3d_llc_data(xx_var, meta)
        recs = np.where(abs(xx.mean(axis=(1,2,3))) > thresh)[0]

        # load xx1
        xx_list = []
        for iternum in optim_iters:
            xx_var = f'xx_{ctrl_var}{ext}'
            meta_filename = '/'.join(meta['filename'].split('/')[:-1])
            #meta_filename = f'{run_dir}/iter{iternum:04d}/{xx_var}'
            meta['filename'] = meta_filename
            meta['filename'] = meta['filename'].replace('iter0000', f'iter{iternum:04d}').replace(f'{0:010d}', f'{iternum:010d}')
            meta['vars'] = xx_var
            xx1 = read_3d_llc_data(xx_var, meta)
            xx1 = xx1.isel(time=recs)
            xx_list.append(xx1)

        # load xx0, with the caveat that it doesnt come with a meta file
        xx_var = f'xx_{ctrl_var}{ext}'
        xx_fname = f'{run_dir}/iter0000/{xx_var}{ext}'
        # using info from last loaded xx, i.e. from for loop above
        meta0 = meta.copy()
        meta0['filename'] = meta['filename'].replace(f'iter{iternum:04d}', 'iter0000').replace(f'{iternum:010d}', f'{0:010d}')
        xx0 = read_3d_llc_data(xx_var, meta0)
        xx0 = xx0.isel(time=recs)
        xx_list.insert(0, xx0)

        da = xr.concat(xx_list, dim='ioptim')

        # change e.g. .effective to _effective
        ds[xx_var.replace('.','_')] = da

    return ds

def load_gentim2d_adxx_ds(run_dir, adxx_var):
    ds = xr.Dataset()
    adxx_path = f'{run_dir}/{adxx_var}'
    iternum = int(run_dir.split('/')[-2][-4:])
    meta = get_aste_file_metadata(adxx_path, iternum=iternum, dtype=np.dtype('>f4'))
    adxx = read_3d_llc_data(adxx_var, meta)
    ds[adxx_var] = adxx
    return ds

def get_ctrl_relative_contributions(
        run_dir,
        grid_dir,
        weight_dir=None,
        ctrl_vars=None,
        verbose=False,
        do_plot=False,
        use_ADJ=False,
        ):


    # Try to intuit ctrl variables for this run
    ctrl_vars = grep_ctrl(field='file', fname = f'{run_dir}/data.ctrl')
    ctrl_vars = [x[3:] for x in ctrl_vars] # remove xx_
    ctrl_weights = grep_ctrl(field='weight', fname = f'{run_dir}/data.ctrl')


    if len(ctrl_vars)*len(ctrl_weights) == 0:
        raise ValueError("Error: One or both of ctrl_vars and ctrl_weights are empty.")

    if weight_dir is None:
        weight_dir = run_dir

    # Load sensitivities from ADJ fields
    if use_ADJ:
        ds_list = []
        for ctrl_var in ctrl_vars:
            if verbose: print(ctrl_var)
            prefix = f'ADJ{ctrl_var}'
            ds = open_astedataset(run_dir, grid_dir=grid_dir, extra_variables=package_state_variables, prefix=prefix)
            if len(ds.data_vars) == 0:
                # As of now, this makes the huge assumption that the variables has the same output frequency as the ADJ variables
                print(f'Warning: ctrl_var {ctrl_var} not loaded using prefix={prefix}\nTrying adxx instead')
                ds = load_gentim2d_adxx_ds(run_dir, adxx_var=f'adxx_{ctrl_var}')
                ds = ds.rename({f'adxx_{ctrl_var}':f'ADJ{ctrl_var}'})
                print(f'Warning: Removing records from adxx_{ctrl_var} to reconcile time axes')
    #            ds = ds.isel(time=slice(None, len(ds_list[-1].time)))
                ds = ds.isel(time=slice(abs(len(ds_list[-1].time) - len(ds.time)), None))
                ds['time'] = ds_list[-1]['time'].values
    
            ds_list.append(ds)
        ds = xr.merge(ds_list).compute()
    
        grid = get_llc_grid(ds, domain='aste')
        
        # If necessary, rotate u/vwind
        if 'uwind' in ctrl_vars:
            uwind, vwind = UEVNfromUXVY(
                ds['ADJuwind'].rename({'i':'i_g'}),
                ds['ADJvwind'].rename({'j':'j_g'}),
                ds,
                grid
            )
    #        uwind, vwind = UEVNfromUXVY(ds['ADJuwind'].rename({'i':'i_g'})[0],ds['ADJvwind'].rename({'j':'j_g'})[0],ds,grid)
            ds['ADJuwind'] = uwind
            ds['ADJvwind'] = vwind
    else: # use adxx
        ds_list = []
        for ctrl_var in ctrl_vars:
            if verbose: print(ctrl_var)
            adxx_var = f'adxx_{ctrl_var}'
            ds = load_gentim2d_adxx_ds(run_dir, adxx_var=adxx_var)
            ds_list.append(ds)
        ds = xr.merge(ds_list).compute()

    # Load weights, compute prior = 1/sqrt(weight)
    
    meta_xc = get_aste_file_metadata(grid_dir+'XC', iternum=0, dtype=np.dtype('>f4'))
    ds_weight = xr.Dataset()
    for ctrl_var, ctrl_weight in zip(ctrl_vars, ctrl_weights):
        weight_fname = f'{weight_dir}/{ctrl_weight}'
        meta = meta_xc.copy()
    
        meta['filename'] = weight_fname
        weight = read_3d_llc_data('XC', meta)[0].compute()
    
        ds_weight[f'{ctrl_var}'] = weight
    
    ds_prior = ds_weight**-.5
    ds_prior = ds_prior.where(np.isfinite(ds_prior), np.nan)


    # compute control relative contributions
    nctrl, nt = (len(ctrl_vars), len(ds[adxx_var].time))
    stdcost_array = np.zeros((nctrl, nt))
    
    for ic, ctrl_var in enumerate(ctrl_vars):
        stdcost_cv = ((ds[f'adxx_{ctrl_var}'] * ds_prior[ctrl_var])**2).sum(axis=(1,2,3))**.5
        stdcost_array[ic, :] = stdcost_cv.values
    stdcost = xr.DataArray(
        stdcost_array,
        dims=['ictrl', 'time'],
        coords={'ictrl': ctrl_vars, 'time': ds[f'adxx_{ctrl_vars[0]}'].time}
    )
    
    xx_relcon = (stdcost / stdcost.sum('ictrl'))
    if do_plot:
        return plot(xx_relcon)
    else:
        return xx_relcon, ds, ds_weight, ds_prior

def plot_relcon(xx_relcon, bar_xoffset = 0.1, cmap='jet', fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    nctrl, nt = xx_relcon.shape
    data = xx_relcon.values
    labels = xx_relcon.ictrl.values
    time = np.arange(nt)
    bottom = np.zeros(nt)
    
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i / (nctrl - 1)) for i in range(nctrl)] 

    # Plot each ictrl component as a stacked bar
    for i in range(nctrl):
        ax.bar(time, data[i], bottom=bottom, label=labels[i], color=colors[i])
        bottom += data[i]
    
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Relative Contribution")
    ax.legend(title="Control Variables", title_fontsize=14, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=20)
    ax.set_title("Relative Control Contributions")
    ax.set_xticks(time)
    ax.tick_params(axis='both', labelsize=14)

    ax.set_xlim(-0.5, nt+.5)
    
    return fig, ax
        
