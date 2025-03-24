import xmitgcm
import os
from smartcables import *
from .dataset import *
from .plot import aste_orthographic
from .calc_barostream import *

aste_tiles = [1, 2, 6, 7, 10, 11]

class NatureRun:
    """Handles loading and resampling of the NR dataset."""
    
    def __init__(self,
            nr_dir = '/work/08381/goldberg/ls6/aste_270x450x180/run_template/input_ecco/smart_phibot/phibot_daily/',
            fld_type='bp',
            fld_fname = None,
            ):
        self.nr_dir = nr_dir
        self.fld_type = fld_type
        self.fld_fname = fld_fname
        self.load_nr_field()

    def load_nr_field(self):
        if self.fld_type == 'bp':
            self._load_nr_bp()
        elif self.fld_type == 'psi':
            self._load_nr_psi()
        else:
            raise ValueError("Unsupported field type")
    
    def _load_nr_bp(self):
        aste_tiles = [1, 2, 6, 7, 10, 11]
        nr_list = []
        for face in aste_tiles:
            bp_face_paths = np.sort(glob.glob(self.nr_dir + f'*face{face:02d}*'))
            nr_bp = xr.open_mfdataset(bp_face_paths)
            nr_list.append(nr_bp)
        nr_bp = xr.concat(nr_list, dim='face')
        nr_bp = nr_bp.rename({'face': 'tile'})
        nr_bp['tile'] = np.arange(len(nr_bp.tile))
        self.fld_full = nr_bp.PhiBot # all possible nr times
        self.fld = None

    def _load_nr_psi(self):
        fld = xr.open_mfdataset(self.nr_dir + self.fld_fname).psi
        self.fld = fld.isel(tile=aste_tiles)
        self.fld['tile'] = np.arange(len(self.fld.tile))

class ForecastModel:
    """Handles loading of the FM dataset."""

    def __init__(self,
                run_dir,
                grid_dir='/work/08381/goldberg/ls6/aste_270x450x180/GRID_noblank_real4/',
                iternums=[0],
                datetimes=None,
                ecco_frequency='day',
                fld_type='bp',
                fm_fld_fname=None,
            ):

        self.run_dir = run_dir
        self.grid_dir = grid_dir
        self.iternums = iternums
        self.datetimes = datetimes
        self.ecco_frequency = ecco_frequency
        self.freq_str = self.ecco_frequency[0]
        self.fld_type = fld_type
        self.fm_fld_fname = fm_fld_fname

        self.load_fm_field()

    def load_fm_field(self):
        """Load the field based on the specified type."""
        if self.fld_type == 'bp':
            self.fld_type_str = 'p_b'
            self._load_fm_bp()
        elif self.fld_type == 'psi':
            self.fld_type_str = '\psi'
            self._load_fm_psi()
        else:
            raise ValueError("Unsupported field type. Choose 'bp' or 'psi'.")

    def _load_fm_bp(self):
        """Load the BP field."""
        self.bpr = BPReader(self.run_dir, iternums=self.iternums, ecco_frequency=self.ecco_frequency)
        fm_bp = self.bpr.ds[f'm_bp{self.ecco_frequency}_anom']

        if self.datetimes is not None:
            fm_bp['time'] = self.datetimes

        self.fld = fm_bp

    def _load_fm_psi(self):
        """Load or generate the PSI field."""

        from pdb import set_trace;set_trace()
        # Generate default filename if not provided
        if self.fm_fld_fname is None:
            date_str = self.datetimes[0].strftime('%Y_%m') if self.datetimes is not None else 'unknown_date'
            self.fm_fld_fname = f'psi_{date_str}_optiters_{self.iternums[-1]}.nc'

        file_path = os.path.join(self.run_dir, self.fm_fld_fname)

        # Check if the file exists
        if os.path.exists(file_path):
            print(f'Loading existing file: {file_path}')
            print('aaaa')
            fld = xr.open_dataset(file_path).psi
        else:
            print(f'File not found: {file_path}')
            print('Loading trsp diagnostics dataset')

            self.ds_trsp = open_asteoptimdataset(
                self.run_dir,
                grid_dir=os.path.join(self.run_dir, f'iter{self.iternums[0]:04d}/'),
                optim_iters=[self.iternums[-1]],
                prefix=['trsp_3d_set1']
            )

            self.ds_trsp = self.ds_trsp.isel(time=slice(0, len(self.datetimes)))
            self.ds_trsp['time'] = self.datetimes

            AB = AsteBarostream(self.ds_trsp, domain='aste', nx=len(self.ds_trsp.i))
            AB.add_to_mygrid()
            AB.calc_barostreams(noDiv=True)
            fld = AB.aste_ds.psi

            # Save to file
            fld.to_netcdf(file_path)
            print(f'Saved generated field to: {file_path}')
        self.fld = fld

class OSSE:
    """Handles the comparison of a single FM against the NR."""
    
    def __init__(self, forecast_model, nature_run, open_asteoptimdataset_kwargs = {}):
        self.fm = forecast_model
        self.nr = nature_run
        
        if self.nr.fld is None or not (
            np.array_equal(self.nr.fld.time.values, self.fm.fld.time.values)
        ):
            self.nr.fld = self.nr.fld_full.sel(time=slice(self.fm.datetimes[0], self.fm.datetimes[-1]))
            self.nr.fld = self.nr.fld.resample(time=f'1{self.fm.freq_str}').mean()

        self.load_grid_ds()
        self.compute_skill()

    def load_grid_ds(self):
        self.grid_ds = open_astedataset(self.fm.grid_dir, grid_dir=self.fm.grid_dir, iters=None)

    def compute_skill(self):
        fld_before = self.fm.fld.isel(ioptim=0).compute()
        fld_after = self.fm.fld.isel(ioptim=-1).compute()
        rms_before = compute_rms(fld_before, self.nr.fld, dim='time')
        rms_after = compute_rms(fld_after, self.nr.fld, dim='time')
        self.fld_skill = (1 - (rms_after / rms_before)).compute()

    def plot_skill(self, vmax_default=0.01, ao_kwargs=None, threshold_skill=None, fig=None, ax=None, **plot_kwargs):
        if ao_kwargs is None:
            ao_kwargs = {}
        return _plot_skill(self, vmax_default=vmax_default, ao_kwargs=ao_kwargs,
                           threshold_skill=threshold_skill, fig=fig, ax=ax, **plot_kwargs)



class MultiOSSE:
    """Facilitates multiple FM-NR comparisons using a fixed NR dataset."""
    
    def __init__(self, nature_run):
        self.nr = nature_run
        self.comparisons = {}

    def add_forecast_model(self, run_dir, **fm_kwargs):
        fm = ForecastModel(run_dir, **fm_kwargs)
        osse = OSSE(fm, self.nr)
        self.comparisons[run_dir] = osse

    def get_skills(self):
        return {run_dir: osse.fld_skill for run_dir, osse in self.comparisons.items()}

def compute_rms(da1, da2, dim, remove_means=True):
    if remove_means:
        da1 = da1 - da1.mean(dim=dim)
        da2 = da2 - da2.mean(dim=dim)
    diff = da1 - da2
    return np.sqrt((diff ** 2).mean(dim=dim))


def _plot_skill(osse, vmax_default=0.01, ao_kwargs=None, threshold_skill=None, fig=None, ax=None, nlev=21, **plot_kwargs):
    if ao_kwargs is None:
        ao_kwargs = {}

    fld_skill = osse.fld_skill
    if threshold_skill is not None:
        fld_skill = fld_skill.where(abs(fld_skill) > threshold_skill)

    if ax is None:
        fig, ax = aste_orthographic(**ao_kwargs)
    
    vmax = plot_kwargs.pop('vmax', vmax_default)
    vmin = -vmax

    cmap = smart_cmaps.custom_div_cmap(nlev, template_cmap=cmocean.cm.curl_r)

    _, ax, cb, p = osse.grid_ds.c.plotpc(fld_skill, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, **plot_kwargs)
    if hasattr(osse.fm, 'bpr'):
        cable_lons, cable_lats = [osse.grid_ds[coord].isel(osse.fm.bpr.sensor_args).values for coord in ['XC', 'YC']]
        ax.scatter(cable_lons, cable_lats, edgecolor='k', s=10, facecolor='w', transform=ccrs.PlateCarree())
    ax.set_title(rf'${osse.fm.fld_type_str}$ skill', fontsize=30, pad=20)
    return fig, ax, cb, p

