# Functions to compute subpolar gyre index in ASTE and ECCO
# Follows this notebook https://github.com/royalosyin/Python-Practical-Application-on-Climate-Variability-Studies/blob/master/ex18-EOF%20analysis%20global%20SST.ipynb
from .tracer import *
import xmitgcm
import ecco_v4_py as ecco
import numpy as np
import xarray as xr
from scipy import signal
from eofs.standard import Eof
import matplotlib.pyplot as plt
import cmocean


def resample_to_latlon(ds, da,
        new_grid_delta_lat = 1/3,
        new_grid_delta_lon = 1/3,    
        new_grid_min_lat = -90,
        new_grid_max_lat = 90,    
        new_grid_min_lon = -180,
        new_grid_max_lon = 180,
    ):
    
#    new_grid_lon_centers, new_grid_lat_centers,\
#    new_grid_lon_edges, new_grid_lat_edges,\
#    field_nearest_1_3rddeg =\
    return ecco.resample_to_latlon(ds.XC, \
                                    ds.YC, \
                                    da,\
                                    new_grid_min_lat, new_grid_max_lat, new_grid_delta_lat,\
                                    new_grid_min_lon, new_grid_max_lon, new_grid_delta_lon,\
                                    fill_value = np.NaN, \
                                    mapping_method = 'nearest_neighbor',
                                    radius_of_influence = 120000)


def compute_spg(ds,
                tslice=slice(None, None),
                new_grid_delta_lat = 1,
                new_grid_delta_lon = 1,
                SPG_mask = None,
                remove_clim = False,
  ):

  da_etan = ds.ETAN.sel(time=tslice)
  nt = len(da_etan)
  new_grid_lon_centers, new_grid_lat_centers, _, _, etan = resample_to_latlon(ds, da_etan,
                                                                             new_grid_delta_lat=new_grid_delta_lat,
                                                                             new_grid_delta_lon=new_grid_delta_lon,
                                                                             )
  msk = ds.hFacC[0]
  _, _, _, _, hfc = resample_to_latlon(ds, msk,
                                      new_grid_delta_lat=new_grid_delta_lat,
                                      new_grid_delta_lon=new_grid_delta_lon,
                                      )
  
  # following the tutorial, they want a -1, 0 mask and that seems to be working
  hfc[np.isnan(hfc)] = 0
  hfc = hfc-1

  dims = ['time', 'lat', 'lon']
  dsll = xr.Dataset(
      data_vars={
          'ETAN': xr.DataArray(etan, dims=dims)
      },
      coords={
          'hFacC': (['lat', 'lon'], hfc),
          'lon': (['lon'], new_grid_lon_centers[0, :]),
          'lat': (['lat'], new_grid_lat_centers[:, 0]), 
          'time': (['time'], da_etan.time.values),
      }
  )

  if SPG_mask is None:
    lon_min, lon_max, lat_min, lat_max = (-60, 10, 40, 65)
    SPG_mask = (dsll.lon >= lon_min) & (dsll.lon <= lon_max) & (dsll.lat >= lat_min) & (dsll.lat <= lat_max)


  lsmask = dsll.hFacC.fillna(0).values
  lsm = np.stack([lsmask]*nt,axis=-1).transpose((2,0,1))
  
  etan_anom = dsll.ETAN - dsll.ETAN.mean('time')

  # subset to SPG region
  etan_anom = etan_anom.where(SPG_mask)
  
#  all_months_have_two = (etan_anom.groupby('time.month')
#                         .count('time')  # Count number of time entries per group
#                         .min() >= 2)    # Check if the minimum count is at least 2
  
  
  if remove_clim:
      climatology = etan_anom.groupby('time.month').mean(dim='time')
      deseasonalized_etan_anom = etan_anom.groupby('time.month') - climatology
      deseasonalized_etan_anom_yr = deseasonalized_etan_anom.groupby('time.year').mean(dim='time')
  else:
      deseasonalized_etan_anom_yr = etan_anom
  
  lsmask = dsll.hFacC.fillna(0).values
  lsm = np.stack([lsmask]*len(deseasonalized_etan_anom_yr),axis=-1).transpose((2,0,1))
  
  deseasonalized_etan_anom_yr = np.ma.masked_array(deseasonalized_etan_anom_yr, mask=lsm)
  deseasonalized_etan_anom_yr[lsm<0] = np.nan
  
  lat = dsll.lat.values
  
  wgts   = np.cos(np.deg2rad(lat))
  wgts   = wgts.reshape(len(wgts), 1)
#  from pdb import set_trace;set_trace() 
#  return SPG_mask, dsll

  solver = Eof(deseasonalized_etan_anom_yr, weights=wgts)
  eof1 = solver.eofs(neofs=10)
  pc1  = solver.pcs(npcs=10, pcscaling=0)
  varfrac = solver.varianceFraction()
  lambdas = solver.eigenvalues()
  return SPG_mask, dsll, eof1, pc1, varfrac, lambdas
