import xmitgcm
from smartcables import *
from .dataset import open_astedataset
from .plot import aste_orthographic

class NatureRun:
    """Handles loading and resampling of the NR dataset."""
    
    def __init__(self,
            bp_nr_dir = '/work/08381/goldberg/ls6/aste_270x450x180/run_template/input_ecco/smart_phibot/phibot_daily/',
            ):
        self.bp_nr_dir = bp_nr_dir
        self.load_nr_bp_anom()

    def load_nr_bp_anom(self):
        aste_tiles = [1, 2, 6, 7, 10, 11]
        nr_list = []
        for face in aste_tiles:
            bp_face_paths = np.sort(glob.glob(self.bp_nr_dir + f'*face{face:02d}*'))
            nr_bp = xr.open_mfdataset(bp_face_paths)
            nr_list.append(nr_bp)
        nr_bp = xr.concat(nr_list, dim='face')
        nr_bp = nr_bp.rename({'face': 'tile'})
        nr_bp['tile'] = np.arange(len(nr_bp.tile))
        self.bp_full = nr_bp.PhiBot # all possible nr times
        self.bp = None

class ForecastModel:
    """Handles loading of the FM dataset."""
    
    def __init__(self,
                run_dir,
                grid_dir = '/work/08381/goldberg/ls6/aste_270x450x180/GRID_noblank_real4/',
                iternums = [0],
                datetimes = None,
                ecco_frequency = 'day',
            ):

        self.run_dir = run_dir
        self.grid_dir = grid_dir
        self.iternums = iternums
        self.datetimes = datetimes
        self.ecco_frequency = ecco_frequency
        self.freq_str = self.ecco_frequency[0]
        self.load_fm_bp_anom()

    def load_fm_bp_anom(self):
        self.bpr = BPReader(self.run_dir, iternums=self.iternums, ecco_frequency=self.ecco_frequency)
        fm_bp = self.bpr.ds[f'm_bp{self.ecco_frequency}_anom']

        if self.datetimes is not None:
            fm_bp['time'] = self.datetimes

        self.bp = fm_bp

class BPOSSE:
    """Handles the comparison of a single FM against the NR."""
    
    def __init__(self, forecast_model, nature_run, open_asteoptimdataset_kwargs = {}):
        self.fm = forecast_model
        self.nr = nature_run
#        self.open_asteoptimdataset_kwargs = open_asteoptimdataset_kwargs
        
        # Check if the currently loaded NR timeframe matches the new FM timeframe
        if self.nr.bp is None or not (
            np.array_equal(self.nr.bp.time.values, self.fm.bp.time.values)
        ):
            self.nr.bp = self.nr.bp_full.sel(time=slice(self.fm.datetimes[0], self.fm.datetimes[-1]))
            self.nr.bp = self.nr.bp.resample(time=f'1{self.fm.freq_str}').mean()

        self.load_grid_ds()
        self.compute_skill()

    def load_grid_ds(self):
        self.grid_ds = open_astedataset(self.fm.grid_dir, grid_dir=self.fm.grid_dir, iters=None)#**self.open_asteoptimdataset_kwargs)

    def compute_skill(self):
        bp_before = self.fm.bp.isel(ioptim=0).compute()
        bp_after = self.fm.bp.isel(ioptim=-1).compute()
        rms_before = compute_rms(bp_before, self.nr.bp, dim='time')
        rms_after = compute_rms(bp_after, self.nr.bp, dim='time')
        self.bp_skill = (1 - (rms_after / rms_before)).compute()

    def plot_skill(self, vmax_default=0.01, ao_kwargs={}, **plot_kwargs):
        fig, ax = aste_orthographic(**ao_kwargs)
        
        vmax = plot_kwargs.pop('vmax', vmax_default)
        vmin = -vmax

        cmap = smart_cmaps.custom_div_cmap(21, template_cmap=cmocean.cm.curl_r)

        _, ax, cb, p = self.grid_ds.c.plotpc(self.bp_skill, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, **plot_kwargs)
        cable_lons, cable_lats = [self.grid_ds[coord].isel(self.fm.bpr.sensor_args).values for coord in ['XC', 'YC']]
        ax.scatter(cable_lons, cable_lats, edgecolor='k', s=10, facecolor='w', transform=ccrs.PlateCarree())
        ax.set_title(r'$p_b$ skill', fontsize=30, pad=20)
        return fig, ax, cb

class MultiBPOSSE:
    """Facilitates multiple FM-NR comparisons using a fixed NR dataset."""
    
    def __init__(self, nature_run):
        self.nr = nature_run
        self.comparisons = {}

    def add_forecast_model(self, run_dir, **fm_kwargs):
        fm = ForecastModel(run_dir, **fm_kwargs)
        bposse = BPOSSE(fm, self.nr)
        self.comparisons[run_dir] = bposse

    def get_skills(self):
        return {run_dir: bposse.bp_skill for run_dir, bposse in self.comparisons.items()}

def compute_rms(da1, da2, dim, remove_means=True):
    if remove_means:
        da1 = da1 - da1.mean(dim=dim)
        da2 = da2 - da2.mean(dim=dim)
    diff = da1 - da2
    return np.sqrt((diff ** 2).mean(dim=dim))

