from smartcables import *

def load_gentim2d_ds(run_dir, ctrl_vars, optim_iters=range(1, 2), thresh=1e5):
    
    ds = xr.Dataset()
    for ctrl_var in ctrl_vars:
        # load adxx(iternum=0) just to get recs
        iternum = 0
        xx_var = f'adxx_{ctrl_var}'
        xx_fname = f'{run_dir}/iter{iternum:04d}/{xx_var}'
        meta = get_aste_file_metadata(xx_fname, iternum=iternum, dtype=np.dtype('>f4'))
        xx = read_3d_llc_data(xx_var, meta)
        recs = np.where(abs(xx.mean(axis=(1,2,3))) > thresh)[0]
        from pdb import set_trace;set_trace();

        # load xx1
        xx_list = []
        for iternum in optim_iters:
            xx_var = f'xx_{ctrl_var}'
            xx_fname = f'{run_dir}/iter{iternum:04d}/{xx_var}'
            meta = get_aste_file_metadata(xx_fname, iternum=iternum, dtype=np.dtype('>f4'))
            xx1 = read_3d_llc_data(xx_var, meta)
            xx1 = xx1.isel(time=recs)
            xx_list.append(xx1)

        # load xx0, with the caveat that it doesnt come with a meta file
        xx_var = f'xx_{ctrl_var}'
        xx_fname = f'{run_dir}/iter0000/{xx_var}'
        # using info from last loaded xx, i.e. from for loop above
        meta0 = meta.copy()
        meta0['filename'] = meta['filename'].replace(f'iter{iternum:04d}', 'iter0000').replace(f'{iternum:010d}', f'{0:010d}')
        xx0 = read_3d_llc_data(xx_var, meta0)
        xx0 = xx0.isel(time=recs)
        xx_list.insert(0, xx0)

        da = xr.concat(xx_list, dim='ioptim')

        ds[xx_var] = da
    return ds
