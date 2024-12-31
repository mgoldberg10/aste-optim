import asteoptim as xs
import xmitgcm
import ecco_v4_py as ecco
import numpy as np
import xarray as xr
from scipy import signal
from eofs.standard import Eof
import matplotlib.pyplot as plt
import cmocean
from asteoptim.spg import *
import warnings
warnings.filterwarnings('ignore')

def load_ds_2012(run_dir, prefix=['state_2d_set1']):
    root_dir = '/nobackup/mgoldbe1/aste_270x450x180/'
    run_dir = root_dir + run_dir

    grid_dir = root_dir + 'GRID_froman/'
    
    ref_date = '2002-01-01 00:00:00'
    delta_t = 1200
    ds = xs.open_asteoptimdataset(run_dir, optim_iters=[0, 1], grid_dir=grid_dir, prefix=prefix,
                                  delta_t=delta_t,ref_date=ref_date)
    time_2012 = slice('2012', '2012')
    ds = ds.sel(time=time_2012)
    return ds

# save plots as pngs, create animations
root_dir = '/nobackup/mgoldbe1/aste_270x450x180/'
subgyre_run_dir = 'osses/run_phibot_fullNR_ndaysrec2_astecoarseweight_subgyre__v4r5obcspretanbl_GiV4r3_nlfs_adv30_580_it0000_pk0000508320/'
fig_dir = root_dir + subgyre_run_dir + 'images/'

# load diagnostics
ds_subgyre_uv = load_ds_2012(subgyre_run_dir, prefix=['trsp_3d_set1'])
ds_subgyre_surf = load_ds_2012(subgyre_run_dir, prefix=['state_2d_set1'])

# compute barotropic streamfunction
AB = xs.AsteBarostream(ds_subgyre_uv, domain='aste')
AB.add_to_mygrid()
from pdb import set_trace;set_trace()
AB.calc_barostreams(noDiv=True)

# assemble ds
ds = AB.aste_ds
ds['bp_anom'] = 100 / 9.81 * (ds_subgyre_surf.PHIBOT - ds_subgyre_surf.PHIBOT.mean('time'))

# loop through fields, set args
das = [ds.psi.compute(), ds.bp_anom.compute()]
cbar_labels = ['[Sv]', '[cm]']
suptitle_pfxs = [r'Barotropic Streamfunction $\psi$ ', r'BP Anomaly $\phi/g$ ']
fld_names = ['psi', 'bp']
fig_dirs = [f'{fig_dir}/{fld_name}/' for fld_name in fld_names]
scale_factors = [2 / 50, 0.2]
vmins = [-50, -10]
vmaxs = [50, 10]

nt = len(ds.time)

for da, cbar_label, suptitle_pfx, fld_name, fig_dir, scale_factor, vmin, vmax in zip(
    das, cbar_labels, suptitle_pfxs, fld_names, fig_dirs, scale_factors, vmins, vmaxs
):
    for time in range(nt):
        da_time = da.isel(time=time)
        mm_yyyy = da_time.time.dt.strftime('%m-%Y').values.item()
        suptitle = suptitle_pfx + f'{mm_yyyy}'

        # make before_after_diff plot
        fig, axes = xs.custom_plot.plot_before_after_diff_rdbu(
            ds_subgyre_uv, da_time,
            am_kwargs = dict(vmin=vmin, vmax=vmax, cbar_label=cbar_label),
            plot_type='am',
            scale_factor = scale_factor,
            suptitle = suptitle,
        )
        fig.set_size_inches(12, 4)
        fig.tight_layout()

        mm_yyyy = mm_yyyy.replace('-', '_')
        fig_path = fig_dir + f'{fld_name}_{mm_yyyy}.png'
        
        fig.savefig(fig_path, dpi=500, bbox_inches='tight')


