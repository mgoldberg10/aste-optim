# Mostly lifted from Tim Smith's (timothyas) pych repository
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.geoaxes import GeoAxesSubplot
import pyresample as pr
from ecco_v4_py.vector_calc import UEVNfromUXVY
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.path as mpath
import matplotlib.colors as mcolors
from smartcables.smart_cmaps import *
import xarray as xr
import copy

@xr.register_dataset_accessor('c')
class GeoAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def ploto(self, da, **am_kwargs):
        fig, ax, cb = plot_orthographic(self._obj, da, **am_kwargs)
        return fig, ax, cb

    def plotpc(self, da, am_init_kwargs=dict(), **am_kwargs):
        fig, ax, cb, p = plot_platecarree(self._obj, da, am_init_kwargs=am_init_kwargs, **am_kwargs)
        return fig, ax, cb, p

def plot_orthographic(ds, da, **am_kwargs):
    fig, ax = aste_orthographic()
    am = aste_map(ds)
    ax, cb, _, _, _ = am(da, ax=ax, **am_kwargs)
    return fig, ax, cb

def plot_platecarree(ds, da, am_init_kwargs=dict(), **am_kwargs):
    ax = am_kwargs.pop('ax', None)
    fig = None if ax else plt.figure(figsize=(8, 6))
    ax = ax or plt.axes(projection=ccrs.PlateCarree())

    am = aste_map(ds, **am_init_kwargs)
    ax, cb, p, _, _ = am(da, ax=ax, **am_kwargs)
    return fig, ax, cb, p


def get_xy_coords(xda):
    """Return the dimension name for x and y coordinates
        e.g. XC or XG
    
    Parameters
    ----------
    xda : xarray DataArray
        with all grid information

    Returns
    -------
    x,y : str
        with e.g. 'XC' or 'YC'
    """

    x = 'XC' if 'XC' in xda.coords else 'XG'
    y = 'YC' if 'YC' in xda.coords else 'YG'
    return x,y

class aste_map:
    # Note that most of this was copied and pasted from the xgcm documentation
    # Because I couldn't write anything fancier

    def __init__(self, ds, dx=0.25, dy=0.25):

        # Extract LLC 2D coordinates
        lons_1d = ds.XC.values.ravel()
        lats_1d = ds.YC.values.ravel()

        # Define original grid
        self.orig_grid = pr.geometry.SwathDefinition(lons=lons_1d, lats=lats_1d)

        # Longitudes latitudes to which we will we interpolate
        lon_tmp = np.arange(-180, 180, dx) + dx / 2
        lat_tmp = np.arange(-35, 90, dy) + dy / 2

        # Define the lat lon points of the two parts.
        self.new_grid_lon, self.new_grid_lat = np.meshgrid(lon_tmp, lat_tmp)
        self.new_grid = pr.geometry.GridDefinition(
            lons=self.new_grid_lon, lats=self.new_grid_lat
        )
        self.ds = ds

    def __call__(
        self,
        da,
        ax=None,
        op=None,
        figsize=(6, 6),
        show_cbar=True,
        cbar_label=None,
        plot_type='pcolormesh',
        addLand=True,
        addBathyContours=False,
        addQuiver=False,
        addStreamlines=False,
        grid=None,
        gridline_kwargs={},
        contour_kwargs={},
        quiver_kwargs={},
        cbar_kwargs={},
        cbar_ticks_params={},
        lon_0 = 80,
        **plt_kwargs,
    ):

        if addStreamlines and addQuiver:
            assert (not (addStreamlines and addQuiver)), 'cannot do both quiver and streamlines'


        tiledim = "tile" if "face" not in da.dims else "face"
        allowed_dims = {"i", "j", "i_g", "j_g"}
        assert (
            tiledim in da.dims and set(da.dims) - {tiledim} <= allowed_dims and len(set(da.dims) - {tiledim}) > 0
        ), f"da must have dimensions [{tiledim}, and at least one of 'i', 'j', 'i_g', or 'j_g']"
    
        field = self.regrid(da)

        vmax = np.nanmax(field)
        vmin = np.nanmin(field)
        if vmax * vmin < 0:
            vmax = np.nanmax([np.abs(vmax), np.abs(vmin)])
            vmin = -vmax
        vmax = plt_kwargs.pop("vmax", vmax)
        vmin = plt_kwargs.pop("vmin", vmin)

        # Handle colorbar and NaN color
        cmap = "RdBu_r" if vmax * vmin < 0 else "viridis"
        cmap = plt_kwargs.pop("cmap", cmap)
        if isinstance(cmap,str):
            cmap = plt.cm.get_cmap(cmap)

        x, y = self.new_grid_lon, self.new_grid_lat

        ## Find index where data is splitted for mapping
        split_lon_idx = round(
#            x.shape[1] / (360 / (op.lon_0 if op.lon_0 > 0 else op.lon_0 + 360))
            x.shape[1] / (360 / (lon_0 if lon_0 > 0 else lon_0 + 360))
        )
        levels = np.linspace(vmin, vmax, 10)
        levels = plt_kwargs.pop('levels', levels)

        transform_first = plt_kwargs.pop('transform_first', True)
        
        if plot_type == 'pcolormesh':
            pl = ax.pcolormesh(
                x[:, :split_lon_idx],
                y[:, :split_lon_idx],
                field[:, :split_lon_idx],
                vmax=vmax,
                vmin=vmin,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                zorder=0,
                **plt_kwargs,
            )
            pr = ax.pcolormesh(
                x[:, split_lon_idx:],
                y[:, split_lon_idx:],
                field[:, split_lon_idx:],
                vmax=vmax,
                vmin=vmin,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                zorder=0,
                **plt_kwargs,
            )
        elif plot_type == 'contourf':
            pl = ax.contourf(
                x[:, :split_lon_idx],
                y[:, :split_lon_idx],
                field[:, :split_lon_idx],
                vmax=vmax,
                vmin=vmin,
                cmap=cmap,
                levels=levels,
                transform=ccrs.PlateCarree(),
                zorder=0,
                transform_first=True,
                extend='both',
                **plt_kwargs,
            )
            pr = ax.contourf(
                x[:, split_lon_idx:],
                y[:, split_lon_idx:],
                field[:, split_lon_idx:],
                vmax=vmax,
                vmin=vmin,
                cmap=cmap,
                levels=levels,
                transform=ccrs.PlateCarree(),
                zorder=0,
                transform_first=True,
                extend='both',
                **plt_kwargs,
            )
        elif plot_type == 'contour':
            pl = ax.contour(
                x[:, :split_lon_idx],
                y[:, :split_lon_idx],
                field[:, :split_lon_idx],
                transform=ccrs.PlateCarree(),
                levels=levels,
                zorder=0,
                **plt_kwargs,
            )
            pr = ax.contour(
                x[:, split_lon_idx:],
                y[:, split_lon_idx:],
                field[:, split_lon_idx:],
                transform=ccrs.PlateCarree(),
                levels=levels,
                zorder=0,
                **plt_kwargs,
            )
        elif plot_type is None:
            pl = pr = None
            pass 
        else:
            ValueError('Please provide valid plot_type, such as \'pcolormesh\', \'contour\', or \'contourf\'')

        if addBathyContours:
            field = self.regrid(self.ds.Depth.where(self.ds.hFacC[0]))

            # Plot each tile separately
            C = ax.contour(
                x[:, :split_lon_idx],
                y[:, :split_lon_idx],
                field[:, :split_lon_idx],
                transform=ccrs.PlateCarree(),
                zorder=1,
                **contour_kwargs,
            )
            C = ax.contour(
                x[:, split_lon_idx:],
                y[:, split_lon_idx:],
                field[:, split_lon_idx:],
                transform=ccrs.PlateCarree(),
                zorder=1,
                **contour_kwargs,
            )

        if addQuiver:
            self.quiver(grid, ax=ax, **quiver_kwargs)
        if addStreamlines:
            streamplot(self.ds, grid,ax=ax, maskW=self.ds.maskW,maskS=self.ds.maskS,
                       scaleByKE=4,ke_threshold=.025,density=2,color='k');

        # Add land and coastlines
        if addLand:
            ax.add_feature(cf.LAND.with_scale("50m"), facecolor='0.75',zorder=3)

        # Add label from attributes
        if cbar_label is None:
            cbar_label = ""
            if "long_name" in da.attrs:
                cbar_label = cbar_label+da.long_name+" "
            if "units" in da.attrs:
                cbar_label = cbar_label+f"[{da.units}]"

        # Colorbar...
        if show_cbar:
            cbar_kwargs = copy.deepcopy(cbar_kwargs)

            # Ensure vmin and vmax exist in cbar_kwargs or extract from the plotted data
            vmin = cbar_kwargs.get("vmin", pl.norm.vmin)
            vmax = cbar_kwargs.get("vmax", pl.norm.vmax)
            pad = cbar_kwargs.pop("pad", 0.1)
            orientation = cbar_kwargs.pop("orientation", 'horizontal')
            default_ticks = np.linspace(vmin, vmax, 5)
            cbar_kwargs.setdefault("ticks", default_ticks)

            if "ticks" not in cbar_kwargs:
                cbar_kwargs["ticks"] = default_ticks
        

            cb = plt.colorbar(
                pl,
                ax=ax,
                shrink=0.8,
                orientation=orientation,
                pad=pad,
                **cbar_kwargs
            )
            cb.set_label(cbar_label)
            cb.ax.tick_params(**cbar_ticks_params)
            cb.set_ticklabels([f"{t:.2f}" for t in cbar_kwargs["ticks"]])
        else:
            cb = None
        return ax, cb, pl, pr, split_lon_idx


    def regrid(self, xda):
        """regrid xda based on llcmap grid"""
        return pr.kd_tree.resample_nearest(
            self.orig_grid,
            xda.values,
            self.new_grid,
            radius_of_influence=100000,
            fill_value=None,
        )

    def quiver(self, grid, ax=None, maskW=None, maskS=None, time=0, ioptim=0, skip=1, diff_ioptim=False, ke_threshold=.1, subset_dict=None, **kwargs):

        if diff_ioptim:
            ds = self.ds.diff('ioptim').copy()
        else:
            ds = self.ds.copy()

        if subset_dict is None:
            subset_dict = dict({'ioptim': ioptim,
                            'time'  : time})

        dims = list(ds.UVELMASS.dims)
        for subset_dim in subset_dict.keys():
            if subset_dim not in dims:
                subset_dict.pop(subset_dim)

        u = ds.UVELMASS.isel(subset_dict)
        v = ds.VVELMASS.isel(subset_dict)

        u = u if maskW is None else u
        v = v if maskS is None else v
        
        u, v = UEVNfromUXVY(u.mean('k'), v.mean('k'), coords=ds, grid=grid)
        
        u,v = [ff.where(ds['maskC'].any('k')) for ff in [u,v]]

        if 'time' in u.dims:
            u, v = u.isel(time=time), v.isel(time=time)

        ke = np.sqrt(u**2 + v**2)
        max_speed = np.nanmax(ke)
        u = u.where(ke>ke_threshold*ke.max(),np.nan)
        v = v.where(ke>ke_threshold*ke.max(),np.nan)
        
        x, y = self.new_grid_lon, self.new_grid_lat
        u = self.regrid(u)
        v = self.regrid(v)
        
        # downsample
        x, y, u, v = [ff[::skip, ::skip] for ff in [x, y, u, v]]
        q = ax.quiver(x, y, u, v, transform=ccrs.PlateCarree(), **kwargs)
        X, Y = .8, .2
        #ax.text(X, Y, f'0.02 m/s', transform=ax.transAxes,
        #        bbox=dict(boxstyle="round,pad=2", edgecolor="black", facecolor="lightgrey"),
        #        fontsize=10, zorder=1)
        #ax.quiverkey(q, X=X+.085, Y=Y-.02, U=0.02, label='', labelpos='E', coordinates='axes', zorder=2)




def quiver(ds,grid,ax=None,maskW=None,maskS=None,skip=1, ke_threshold=.1, **kwargs):
    """
    Make a quiver plot from velocities in a dataset

    Parameters
    ----------
    ds : xarray Dataset
        with UVELMASS, VVELMASS fields
    grid : xgcm Grid object
        to interp with
    ax : matplotlib axis object, optional
        defining the axis to plot on
    maskW, maskS : xarray DataArrays
        selecting the field to be plotted
    skip : int, optional
        how many velocity points to skip by when plotting the arrows
    ke_threshold : float, optional
        fractional of max(np.sqrt(u**2 + v**2)) to plot, removes tiny dots where velocity
        is near zero
    **kwargs : dict, optional
        passed to matplotlib.pyplot.quiver, some common ones are:

        scale : int, optional
            to scale the arrows by
        width : float, optional
            width of the arrowhead
        alpha : float, optional
            opacity of the arrows
    """
    if ax is None:
        _,ax = plt.subplots()
    sl = slice(None,None,skip)
    x = ds.XC.values.flatten()[sl]
    y = ds.YC.values.flatten()[sl]

    u = ds.UVELMASS if maskW is None else ds.UVELMASS.where(maskW)
    v = ds.VVELMASS if maskS is None else ds.VVELMASS.where(maskS)

    u,v = grid.interp_2d_vector({'X':u.mean('Z'),'Y':v.mean('Z')},boundary='fill').values()
    # hard coding - select time=0
    u,v = [ff.where(ds['maskC'].any('Z'))[sl,sl] for ff in [u,v]]

    # hide the little dots where velocity ~0
    ke = np.sqrt(u**2+v**2)
    u = u.where(ke>ke_threshold*ke.max(),np.nan)
    v = v.where(ke>ke_threshold*ke.max(),np.nan)

    # quiver wants numpy arrays
    x,y,u,v = [ff.values for ff in [x,y,u,v]]
    if isinstance(ax,GeoAxesSubplot):
        ax.quiver(x,y,u,v,pivot='tip',transform=ccrs.PlateCarree(),**kwargs)
    else:
        ax.quiver(x,y,u,v,pivot='tip',**kwargs)
    return ax

def streamplot(ds, grid, ax=None, maskW=None, maskS=None, ke_threshold=.1,
               scaleByKE=0, **kwargs):
    """
    Make a quiver plot from velocities in a dataset

    Parameters
    ----------
    ds : xarray Dataset
        with UVELMASS, VVELMASS fields
    grid : xgcm Grid object
        to interp with
    ax : matplotlib axis object, optional
        defining the axis to plot on
    maskW, maskS : xarray DataArrays
        selecting the field to be plotted
    skip : int, optional
        how many velocity points to skip by when plotting the arrows
    ke_threshold : float, optional
        fractional of max(np.sqrt(u**2 + v**2)) to plot, removes tiny dots where velocity
        is near zero
    scaleByKE : float, optional
        if >0, then scale linewidth by this * KE/max(KE)
    **kwargs : dict, optional
        passed to matplotlib.pyplot.quiver, some common ones are:

        density : float, optional
            density of the contours
        linewidth : float, optional
            width of the streamlines, if array same size as u/v, scales with values
        alpha : float, optional
            opacity of the arrows
    """
    if ax is None:
        _,ax = plt.subplots()
    x = ds.XC
    y = ds.YC.interp(YC=np.linspace(ds.YC.min(),ds.YC.max(),len(ds.YC)))

    u = ds.UVELMASS if maskW is None else ds.UVELMASS.where(maskW)
    v = ds.VVELMASS if maskS is None else ds.VVELMASS.where(maskS)

    u,v = grid.interp_2d_vector({'X':u.mean('Z'),'Y':v.mean('Z')},boundary='fill').values()
    u,v = [ff.where(ds['maskC'].any('Z')) for ff in [u,v]]

    # hide the little dots where velocity ~0
    ke = np.sqrt(u**2+v**2)
    u = u.where(ke>ke_threshold*ke.max(),np.nan)
    v = v.where(ke>ke_threshold*ke.max(),np.nan)

    # quiver wants numpy arrays
    if scaleByKE is not None:
        kwargs = {} if kwargs is None else kwargs
        #if kwargs is not None:
        #    #assert 'linewidth' not in kwargs.keys(), \
        #    #        'do not pass linewidth if scaling by KE'
        #    pass
        #else:
        #    kwargs={}

        if isinstance(scaleByKE,(list,tuple)):
            key = scaleByKE[0]
            ratio_factor = scaleByKE[1]
        else:
            key = 'linewidth'
            ratio_factor = scaleByKE

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scaling = xr.where(ke>0,ratio_factor*ke/ke.max(),0.).values

        kwargs[key] = scaling

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x,y,u,v,ke = [ff.values for ff in [x,y,u,v,ke]]
    if isinstance(ax,GeoAxesSubplot):
        ax.streamplot(x,y,u,v,transform=ccrs.PlateCarree(),**kwargs)
    else:
        ax.streamplot(x,y,u,v,**kwargs)
    return ax


def aste_orthographic(subplot_n = 1,
                      subplot_m = 1,
                      xmin = -100,
                      xmax = 30,
                      ymin = 0,
                      ymax = 80,
                      n = 20,
                      figsize = None,
                      landfacecolor='silver',
                      manual_hide_gl_labels=True,
                      ):
    # https://stackoverflow.com/questions/75586978/cartopy-labels-not-appearing-for-high-latitude-non-rectangular-projection/75587005#75587005
    # Create a figure with n rows and m columns of subplots
    if figsize is None:
        figsize = (10 * subplot_m, 6 * subplot_n)
    fig, axes = plt.subplots(subplot_n, subplot_m, figsize=figsize, 
                             subplot_kw={'projection': ccrs.Mollweide(central_longitude=(xmin + xmax) / 2)})

    # Handle the case when there is only one subplot
    if subplot_n == 1 and subplot_m == 1:
        axes = np.array([axes])  # Make axes an array for consistency in further processing

    aoi = mpath.Path(
        list(zip(np.linspace(xmin,xmax, n), np.full(n,ymax))) + \
        list(zip(np.full(n,xmax), np.linspace(ymax,ymin, n))) + \
        list(zip(np.linspace(xmax,xmin, n), np.full(n,ymin))) + \
        list(zip(np.full(n,xmin), np.linspace(ymin,ymax, n)))
    )
    for ax in axes.ravel():    
        ax.set_boundary(aoi, transform=ccrs.PlateCarree())
        
        # Colored Land Background
        land = cf.NaturalEarthFeature('physical','land',scale='110m',facecolor=landfacecolor,lw=1,linestyle='--', zorder=1)
        ax.add_feature(land)
        
    
        ax.set_extent([xmin,xmax,0,90],crs=ccrs.PlateCarree())
        ax.set_extent([xmin,xmax,ymin,ymax],crs=ccrs.PlateCarree())
        # Set gridlines to variable so you can manipulate them
        gl = ax.gridlines(draw_labels=True,crs=ccrs.PlateCarree(),x_inline=False,y_inline=False, linestyle=':', zorder=2)
#        gl.xlocator = mticker.FixedLocator([-100, -80, -60, -40, -20, 0, 20])
        gl.xlocator = mticker.FixedLocator(np.linspace(-180, 180, 19))
#        gl.ylocator = mticker.FixedLocator(range(0, 90, 20))
        gl.ylocator = mticker.FixedLocator(np.linspace(-90, 90, 10))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        
        fig.canvas.draw()

        # can't figure out how to hide right latitude lines. See https://stackoverflow.com/questions/75597673/hide-right-side-axis-latitude-labels-for-high-latitude-non-rectangular-project
        if manual_hide_gl_labels:
            # Step 1: Collect all y-values for labels containing 'W', 'E', or '0°'
            y_values = []
            for label in gl._labels:
                text = label.artist.get_text()
                if 'W' in text or 'E' in text or text == '0°':
                    y_values.append(label.artist.get_position()[1])  # Get the y-position of the label
        
            # Step 2: Get unique y-values
            unique_y_values = list(set(y_values))
        
            # Step 3: Identify the top labels based on the unique y-values
            for label in gl._labels:
                text = label.artist.get_text()
                x, y = label.artist.get_position()
        
                if 'W' in text or 'E' in text or text == '0°':
                    # Determine if the label corresponds to a top y-value
                    is_top = y in unique_y_values and y == max(unique_y_values)
        
                    if is_top:
                        label.artist.set_visible(False)
    if subplot_n == 1 and subplot_m == 1:
        axes = axes[0]
    return fig, axes, gl

