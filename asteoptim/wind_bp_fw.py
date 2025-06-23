import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch
import cartopy.crs as ccrs
from .utils import get_fH
from smartcables.mds_utils import get_aste_file_metadata, read_3d_llc_data
from smartcables.curl import curl
import ecco_v4_py as ecco
from .dataset import open_asteoptimdataset
from .plot import aste_orthographic, aste_map
from smartcables.smart_cmaps import SMARTColormaps


class WindBPFWPlotter:
    def __init__(self, osse):
        self.osse = osse
        self.iternums = osse.fm.iternums
        self.run_dir = osse.fm.run_dir
        self.grid_dir = osse.fm.grid_dir
        self.f, self.H = get_fH(osse.grid_ds)
        self.f_over_H = self.f / self.H
        self.ds_ctrl = None
        self.ds_ctrl_diff = None
        self.uE = None
        self.vN = None

    def load_ds_ctrl(self, run_dir=None, ext='.effective', ctrl_vars=['uwind', 'vwind']):
        run_dir = run_dir or self.run_dir
        ds_ctrls = []
        for iternum in self.iternums:
            ds_ctrl = xr.Dataset()
            for ctrl_var in ctrl_vars:
                ds_ctrl[ctrl_var] = self.load_adxx(run_dir, ctrl_var, iternum, ext)['xx']
            ds_ctrls.append(ds_ctrl)
        self.ds_ctrl = xr.concat(ds_ctrls, dim='ioptim')
        self.ds_ctrl_diff = self.ds_ctrl.diff('ioptim').isel(ioptim=0)

    def load_adxx(self, run_dir, ctrl_var, iternum, ext='.effective'):
        run_dir = run_dir or self.run_dir
        xx_var = f'xx_{ctrl_var}'
        xx_fname = f'{run_dir}/iter{iternum:04d}/{xx_var}{ext}'
        meta = get_aste_file_metadata(xx_fname, iternum=iternum, dtype=np.dtype('>f4'))
        xx = read_3d_llc_data(xx_var, meta)
        return xr.Dataset(data_vars={'xx': xr.DataArray(xx)})

    def load_bp(self, run_dir=None, grid_dir=None):
        run_dir = run_dir or self.run_dir
        grid_dir = grid_dir or self.grid_dir
        ds = open_asteoptimdataset(run_dir, grid_dir=grid_dir, prefix=['state_2d_set1'], optim_iters=self.iternums)
        dsd = ds.diff('ioptim').isel(ioptim=0)
        bp = 100 / 9.81 * dsd.PHIBOT.mean('time')  # Convert from geopotential to pressure anomaly
        return bp.where(self.osse.grid_ds.hFacC[0])

    def load_vel(self, run_dir=None, grid_dir=None):
        run_dir = run_dir or self.run_dir
        grid_dir = grid_dir or self.grid_dir
        self.ds_vel = open_asteoptimdataset(run_dir, grid_dir=grid_dir, prefix=['trsp_3d_set1'], optim_iters=self.iternums)
        self.ds_vel_diff = self.ds_vel.diff('ioptim').isel(ioptim=0).mean('time')
        UVELMASS = self.ds_vel_diff.UVELMASS.compute()
        VVELMASS = self.ds_vel_diff.VVELMASS.compute()
        self.grid = ecco.get_llc_grid(self.ds_vel, domain='aste')
        self.uE, self.vN = ecco.vector_calc.UEVNfromUXVY(UVELMASS, VVELMASS, self.ds_vel_diff, self.grid)

    def load_state3d(self, run_dir=None, grid_dir=None):
        run_dir = run_dir or self.run_dir
        grid_dir = grid_dir or self.grid_dir
        self.ds_state3d = open_asteoptimdataset(run_dir, grid_dir=grid_dir, prefix=['state_3d_set1'], optim_iters=self.iternums)
        self.ds_state3d_diff = self.ds_state3d.diff('ioptim').isel(ioptim=0).mean('time')

    def scatter_sensors_scaled(self, ax, min_size=10, max_size=30):
        osse = self.osse
        ds = osse.fm.bpr.ds
        ds_grid = osse.grid_ds
        cable_lons, cable_lats = [ds_grid[coord].isel(osse.fm.bpr.sensor_args).values for coord in ['XC', 'YC']]
        cost = ds.bpdifanom_smooth**2 * ds.weight
        s_costcontrib = cost.diff('ioptim')[0].sum('time').isel(osse.fm.bpr.sensor_args).values
        s = np.nan_to_num(-s_costcontrib, nan=0.0, posinf=0.0, neginf=0.0)
        s = np.clip(s, a_min=0, a_max=None)
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-8)
        marker_sizes = min_size + s_norm * (max_size - min_size)
        return ax.scatter(cable_lons, cable_lats, s=marker_sizes, transform=ccrs.PlateCarree(), color='black', edgecolor='none')

    def plot_bp(self, bp_da, ax=None, vmin=-0.1, vmax=0.1, nlev=20, **kwargs):
        """
        Plot bottom pressure anomaly using precomputed `bp_da`.
        """
        if ax is None:
            fig, ax = aste_orthographic(projection='LambertConformal', ymin=50, xmin=-80, xmax=-5)
        else:
            fig = None

        cmap = SMARTColormaps(nlev).custom_div_cmap()
        plot_kwargs = dict(vmin=vmin, vmax=vmax, cmap=cmap, levels=np.linspace(vmin, vmax, nlev), plot_type='contourf', cbar_kwargs=dict(pad=.12))
        plot_kwargs.update(kwargs)

        am = aste_map(self.osse.grid_ds)
        ax, cb, p, _, _ = am(bp_da, ax=ax, **plot_kwargs)
        ax.set_title(r'$\delta$(OBP)', fontsize=16)
        cb.ax.set_xlabel('[cm]', fontsize=16)

        ax.coastlines()

        return fig, ax, cb, p

    def get_quiver(self, uda, vda, ke_threshold=0.1, skip=6):
        ke = np.sqrt(uda**2 + vda**2)
        u = uda.where(ke > ke_threshold * ke.max())
        v = vda.where(ke > ke_threshold * ke.max())
        am = aste_map(self.osse.grid_ds)
        x, y = am.new_grid_lon, am.new_grid_lat
        u, v = am.regrid(u), am.regrid(v)
        return [arr[::skip, ::skip] for arr in [x, y, u, v]]

    def plot_quivers(self, ax=None, field='ctrl', scale_ctrl=1e-7, scale_vel=8e-9, ke_threshold=0.1, get_quiver_kwargs={}, **kwargs):
        """
        Plot one quiver field: either 'ctrl' (control difference) or 'phys' (physical velocity).
    
        Parameters:
            ax : matplotlib axis or None
                Axis to plot on. If None, creates a new plot.
            field : str
                Which quiver to plot: 'ctrl' or 'phys'.
            scale_ctrl : float
                Scale factor for control quiver (used only if field='ctrl').
            scale_vel : float
                Scale factor for physical quiver (used only if field='phys').
            ke_threshold : float
                Kinetic energy threshold for masking small vectors.
        """
        if field == 'ctrl' and self.ds_ctrl_diff is None:
            raise ValueError("Control difference data not loaded. Please run `load_ds_ctrl` first.")
        if field == 'phys' and (self.uE is None or self.vN is None):
            raise ValueError("Physical velocity data not loaded. Please run `load_vel` first.")
    
        if ax is None:
            fig, ax = aste_orthographic(projection='LambertConformal', ymin=50, xmin=-80, xmax=-5)
        else:
            fig = None
    
        am = aste_map(self.osse.grid_ds)
        da_empty = xr.full_like(self.osse.grid_ds.Depth, fill_value=np.nan)
        plot_kwargs = dict(vmin=-1, vmax=1, cmap='coolwarm', levels=10, plot_type='contourf', show_cbar=False)
        ax, _, _, _, _ = am(da_empty, ax=ax, **plot_kwargs)
        ax.coastlines()
    
        if field == 'ctrl':
            uda_ctrl = self.ds_ctrl_diff.uwind.mean('time')
            vda_ctrl = self.ds_ctrl_diff.vwind.mean('time')
            ax.quiver(*self.get_quiver(uda_ctrl, vda_ctrl, ke_threshold, **get_quiver_kwargs),
                      transform=ccrs.PlateCarree(), scale=scale_ctrl, scale_units='x', **kwargs)
        elif field == 'phys':
            uda_phys = self.uE[0]
            vda_phys = self.vN[0]
            ax.quiver(*self.get_quiver(uda_phys, vda_phys, ke_threshold, **get_quiver_kwargs),
                      transform=ccrs.PlateCarree(), scale=scale_vel, scale_units='x', **kwargs)
        else:
            raise ValueError("Invalid field requested. Choose 'ctrl' or 'phys'.")
    
        return fig, ax

    def get_relative_quiver_label(self, ax, label_value=0.05, scale=1e-7,
                                  x0=.05, y0=0.08, length_axes=0.07,
                                  nspace=26, box_padx=-.07, box_pady=0.):
        """
        Draws a labeled reference quiver arrow in axes coordinates.
    
        Parameters:
            ax : matplotlib Axes
                Axes to draw on.
            label_value : float
                The value in m/s that the reference arrow should represent.
            scale : float
                The quiver scale used in plot_quivers.
            x0, y0 : float
                Start position of the arrow in axes coordinates.
            length_axes : float
                Visual length of the arrow in axes coords. Optional override.
            nspace : int
                Width of white box behind label.
            box_padx, box_pady : float
                Offset for background box.
        """
        # Compute arrow length in axes coordinates
        # quiver length = magnitude / scale → so this is just a representation
        # You can skip this if you're happy using fixed visual length (`length_axes`)
        arrow_length = length_axes
    
        # Add arrow manually as a patch
        arrow = FancyArrowPatch((x0, y0), (x0 + arrow_length, y0),
                                transform=ax.transAxes,
                                color='black', linewidth=1.,
                                arrowstyle='-|>,head_length=.3,head_width=.1',
                                mutation_scale=10, zorder=5)
        ax.add_patch(arrow)
    
        # Add label
        ax.text(x0 + arrow_length + 0.01, y0, f'{label_value:.2f} m/s',
                transform=ax.transAxes,
                fontsize=11, verticalalignment='center', horizontalalignment='left',
                bbox=dict(facecolor='none', edgecolor='none', boxstyle='round,pad=0.2'),
                zorder=6)
    
        # White background box
        ax.text(x0 + arrow_length + box_padx, y0 + box_pady, ' ' * nspace,
                transform=ax.transAxes,
                fontsize=11, verticalalignment='center', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.4'),
                zorder=4)
    
        return ax

    def plot_f_over_H_contours(self, ax, fH_levels=None, dx=0.5, dy=0.5, **kwargs):
        """
        Plot contours of f/H on the given axis.

        Parameters:
            ax : matplotlib axis
                Axis to plot on.
            fH_levels : array-like, optional
                Contour levels for f/H.
            dx, dy : float
                Grid spacing for the plot.
            **kwargs : additional keyword arguments passed to aste_map plotting.
        """
        if fH_levels is None:
            fH_levels = np.linspace(0, 3e-6, 10)
        fH_kwargs = dict(
            plot_type='contour',
            levels=fH_levels,
            linewidths=0.1,
            colors='k',
            show_cbar=False
        )
        fH_kwargs.update(kwargs)  # allow overrides

        am = aste_map(self.osse.grid_ds, dx=dx, dy=dy)
        ax, _, _, _, _ = am(self.f_over_H, ax=ax, **fH_kwargs)
        return ax



    def get_curl(self, rho_air = 1.225, Cd = 1.3e-3):
        grid = ecco.get_llc_grid(self.osse.grid_ds, domain='aste')
        self.curl = rho_air * Cd * curl(self.ds_ctrl_diff.uwind.mean('time'), self.ds_ctrl_diff.vwind.mean('time'), self.osse.grid_ds, grid)

    def plot_curl(self, ax, fac=1e9):

        def format_cbar_label(fac, unit="Nm$^{-3}$"):
            exponent = int(np.log10(fac))
            base = int(fac / (10**exponent))
            if base == 1:
                return rf"[$10^{{{ -exponent }}}${unit}]"
            else:
                return rf"[$ {base} \times 10^{{{ -exponent }}}${unit}]"
        nlev=16;vmax = 4;vmin = -vmax;levels=np.linspace(vmin, vmax, nlev)
        cmap = SMARTColormaps(nlev).custom_div_cmap()
        pk=dict(vmin=vmin, vmax=vmax, cmap=cmap, levels=levels, plot_type='contourf', cbar_kwargs = dict(pad=.12))

        am = aste_map(self.osse.grid_ds)
        ax, cb, _,_,_ = am(self.curl*fac, ax=ax,**pk)
        cbar_label = format_cbar_label(fac)
        cb.ax.set_xlabel(cbar_label, fontsize=20)


        cb.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        cb.ax.tick_params(labelsize=14)
        return ax
        

    def get_advfw(self, Sref = 34.8, k=0):
        if not hasattr(self, 'ds_vel'):
            raise ValueError("Velocity difference data not loaded. Please run `load_ds_vel` first.")
        if not hasattr(self, 'ds_state3d'):
            raise ValueError("state_3d difference data not loaded. Please run `load_state3d` first.")

        ds_state3d_diff = self.ds_state3d_diff.isel(k=k)
        ds_vel_diff = self.ds_vel_diff.isel(k=k)

        S_at_u = self.grid.interp(ds_state3d_diff.SALT, 'X', boundary='extend')
        S_at_v = self.grid.interp(ds_state3d_diff.SALT, 'Y', boundary='extend')
        
        ADVx_FW = (ds_vel_diff.UVELMASS * ds_vel_diff.dyG * ds_vel_diff.drF * (Sref - S_at_u)/Sref ).compute()
        ADVy_FW = (ds_vel_diff.VVELMASS * ds_vel_diff.dxG * ds_vel_diff.drF * (Sref - S_at_v)/Sref ).compute()

        ADVx_FW_ts = ADVx_FW.sum(ds_vel_diff.hFacW.dims)
        ADVy_FW_ts = ADVy_FW.sum(ds_vel_diff.hFacS.dims)
        ADV_FW = ADVx_FW_ts + ADVy_FW_ts
        self.ADVe_FW, self.ADVn_FW = ecco.vector_calc.UEVNfromUXVY(ADVx_FW, ADVy_FW, ds_vel_diff, self.grid)

