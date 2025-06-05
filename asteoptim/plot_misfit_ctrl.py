import matplotlib.ticker as ticker
import numpy as np
from smartcables.smart_cmaps import *
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from .plot import aste_orthographic

class PlotMisfitCtrl:
    def __init__(
            self,
            bp_forecastmodel,
            grid_ds,
            ctrl_ds,
            plot_times=[0],
            time_mean=False,
            dir_out='',
            verbose=False,
            nlevels=11,
            savefig=False,
            misfit_kwargs=None,
            ctrl_kwargs=None,
    ):
        self.fm = bp_forecastmodel
        self.grid_ds = grid_ds
        self.ctrl_ds = ctrl_ds
        self.plot_times = plot_times
        self.time_mean = time_mean
        self.dir_out = dir_out
        self.verbose = verbose
        self.nlevels = nlevels
        self.savefig = savefig

        self.misfit_kwargs = misfit_kwargs if misfit_kwargs else {"vmin": -5e-3, "vmax": 5e-3, "cmap": custom_div_cmap(nlevels), "show_cbar":False}
        self.ctrl_kwargs = {"cmap": gwp_div_cmap(nlevels), "vmin": -1e-2, "vmax": 1e-2, "show_cbar":False} if ctrl_kwargs is None else {**{"cmap": gwp_div_cmap(nlevels)}, **ctrl_kwargs, "show_cbar":False}

        self.misfit = self.fm.bpr.ds.bpdifanom_smooth

    def __call__(self):
        time_strs = self.misfit.time.dt.strftime('%Y-%m-%d').values
        time_strs = time_strs[self.plot_times]
        cable_lons, cable_lats = [self.grid_ds[coord].isel(self.fm.bpr.sensor_args).values for coord in ['XC', 'YC']]

        if self.time_mean:
            print('WARNING: time_mean of bpdifanom_smooth is zero, plot will be empty')
            misfit_data = self.misfit.mean(dim='time')
            ctrl_uwind = self.ctrl_ds.xx_uwind.mean(dim='time')
            ctrl_vwind = self.ctrl_ds.xx_vwind.mean(dim='time')
            time_strs = ["Time Mean"]
        else:
            misfit_data = self.misfit
            ctrl_uwind = self.ctrl_ds.xx_uwind
            ctrl_vwind = self.ctrl_ds.xx_vwind
            time_strs = self.misfit.time.dt.strftime('%Y-%m-%d').values[self.plot_times]
        # grab ioptim = -1, i.e. after optim
        misfit_data = misfit_data[-1]

        for time, time_str in enumerate(time_strs):
            if self.verbose: print(f'Plotting time {time_str}')

            fig, axes = aste_orthographic(subplot_n=1, subplot_m=3)

            # Misfit plot
            ax = axes[0]
            ticks = np.linspace(self.misfit_kwargs["vmin"], self.misfit_kwargs["vmax"], 5)
#            from pdb import set_trace;set_trace()
            _, ax, cb, p = self.grid_ds.c.plotpc(misfit_data if self.time_mean else misfit_data[time], ax=ax, **self.misfit_kwargs)

            ax.scatter(cable_lons, cable_lats, edgecolor='k', s=10, facecolor='w', transform=ccrs.PlateCarree())
            ax.set_title(r'$\mathcal{S}(m-d)$', fontsize=40, pad=20)

            cbar_ax = fig.add_axes([0.05, 0., 0.28, 0.04])
            fig.colorbar(p, cax=cbar_ax, orientation='horizontal')
            cbar_ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.e'))
            cbar_ax.tick_params(size=20, labelsize=30)
            cbar_ax.set_xlabel('cm', size=30)
            cbar_ax.xaxis.set_ticks(ticks)

            pos = axes[0].get_position()
            new_pos = [pos.x0 - 0.05, pos.y0, pos.width, pos.height]
            axes[0].set_position(new_pos)

            # Control plots
            for i, (data, title) in enumerate(zip([ctrl_uwind, ctrl_vwind], ['xx_uwind', 'xx_vwind']), start=1):
                ax = axes[i]
                _, ax, _, p = self.grid_ds.c.plotpc(
                    data if self.time_mean else data[time], ax=ax, **self.ctrl_kwargs
                )
                ax.scatter(cable_lons, cable_lats, edgecolor='k', s=10, facecolor='w', transform=ccrs.PlateCarree())
                ax.set_title(title, fontsize=30 if i == 0 else 40, pad=20)

            cbar_ax = fig.add_axes([0.4, 0., 0.5, 0.04])
            fig.colorbar(p, cax=cbar_ax, orientation='horizontal')
            cbar_ax.tick_params(size=20, labelsize=30)
            cbar_ax.set_xlabel('m/s', size=30)
            ticks = np.linspace(self.ctrl_kwargs["vmin"], self.ctrl_kwargs["vmax"], 5)
            cbar_ax.xaxis.set_ticks(ticks)

            fig.suptitle(time_str, fontsize=60, y=1.2, x=.45)

            if self.savefig:
                fout = f"{self.dir_out}misfit_and_uvwind_{time_str.replace('-', '_')}.png"
                fig.savefig(fout, dpi=500, bbox_inches='tight')

        return fig, axes


