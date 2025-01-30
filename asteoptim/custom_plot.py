from .plot import *
import cmocean

def plot_aste_rdbu(ds, da, fig=None, ax=None, title='', **am_kwargs):
    if ax is None:
        fig, ax = aste_orthographic(subplot_n=1, subplot_m=1)
    am = aste_map(ds)
    nlev = 21
    cmap = am_kwargs.pop('cmap', cmocean.cm.balance)
   
    vmin = am_kwargs.pop('vmin', -1)
    vmax = am_kwargs.pop('vmax', 1)

    levels = am_kwargs.pop('levels', np.linspace(vmin, vmax, nlev+1))
    extend = am_kwargs.pop('extend', 'both')

    cbar_ticks = np.linspace(vmin, vmax, 5)
    ax, cb, _,_,_ = am(da,
                       ax=ax,
                       do_pcolor=False, 
                       cbar_ticks_params = dict(labelsize=14),
                       cmap=cmap,
                       vmin=vmin,
                       vmax=vmax,
                       levels=levels,
                       extend=extend,
                       **am_kwargs
                    )
    cb.set_ticks(cbar_ticks)
#    cb.extend='both'
    ax.set_title(title, fontsize=20, pad=16)
    return fig, ax, cb

def plot_before_after_diff_rdbu(
                              ds,
                              da,
                              ioptim_before=0,
                              ioptim_after=-1,
                              titles = None,
                              am_kwargs = {},
                              plot_type='at',
                              scale_factor = 0.1,
                              suptitle = '',
                          ):
    # da provided should correspond to a fixed time,
    # i.e. should have spatial dimensions and ioptim dimension

    da_before = da.isel(ioptim=ioptim_before)
    da_after = da.isel(ioptim=ioptim_after)
    da_diff = da_after - da_before
    das = [da_before, da_after, da_diff]
     
    if titles is None:
        titles = ['before optim', 'after optim', 'difference']
    elif len(titles) != 3:
        ValueError('List of titles should have length=3')


    if plot_type == 'am': # fancy astemaps cartopy
      fig, axes = aste_orthographic(subplot_n=1, subplot_m=3)
      for ida, (da, ax, title) in enumerate(zip(das, axes, titles)):
          if ida == 2:
              am_kwargs['vmin']=scale_factor * am_kwargs['vmin']
              am_kwargs['vmax']=scale_factor * am_kwargs['vmax']
          ax, cb = plot_aste_rdbu(ds, da, ax, title, **am_kwargs)
          if abs(am_kwargs['vmax'])*abs(am_kwargs['vmin']) < 1:
              cb.ax.set_xticklabels([f'{i:.1f}' for i in cb.get_ticks()])
    elif plot_type == 'at': # aste tracer view 
      fig, axes = plt.subplots(1, 3)
      for ida, (da, ax, title) in enumerate(zip(das, axes, titles)):
          if ida == 2:
              am_kwargs['vmin']=int(scale_factor * am_kwargs['vmin'])
              am_kwargs['vmax']=int(scale_factor * am_kwargs['vmax'])
          da_at = da.at()
          ax = plot_aste_tracer_rdbu(da_at, ax, title, **am_kwargs)
    else: 
        ValueError(f'Unsupported plot_type \'{plot_type}\'')
    
    fig.suptitle(suptitle, fontsize=30, y=1.)
    fig.set_size_inches(12, 4)
    fig.tight_layout()
    return fig, axes

def plot_aste_tracer_rdbu(da, ax, title='', **am_kwargs):
    kwargs = get_rdbu_plot_defaults(am_kwargs=am_kwargs)
    combined_kwargs = {**kwargs, **am_kwargs}
    da.plot.contourf(ax=ax, **combined_kwargs)
    ax.set_title(title)
    return ax


def get_rdbu_plot_defaults(am_kwargs=None, nlev=20):
    """
    Returns default plotting parameters for the functions.
    
    Parameters:
        am_kwargs (dict, optional): Keyword arguments for the plotting function.
        nlev (int, optional): Number of levels for the contour plot.
        
    Returns:
        dict: A dictionary containing 'nlev', 'cmap', 'vmin', 'vmax', 'levels', and 'cbar_ticks'.
    """
    if am_kwargs is None:
        am_kwargs = {}
    
    # Set default colormap and value range
    cmap = cmocean.cm.balance
    vmin = am_kwargs.pop('vmin', -1)
    vmax = am_kwargs.pop('vmax', 1)
    
    # Generate levels and ticks
    levels = np.linspace(vmin, vmax, nlev + 1)
    cbar_ticks = np.linspace(vmin, vmax, 5)
    
    return {
        'nlev': nlev,
        'cmap': cmap,
        'vmin': vmin,
        'vmax': vmax,
        'levels': levels,
        'cbar_ticks': cbar_ticks
    }

