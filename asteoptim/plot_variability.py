from smartcables import *
import numpy as np
import xarray as xr
import pandas as pd

def plot_jra_vs_aste_cable_variability(
        jra_std,
        ds_ctrl,
        sensor_args,
        grid_ds,
        fig = None,
        ax = None,
        ):

    if fig is None:
        fig, ax = plt.subplots()

    cable_lons, cable_lats = [grid_ds[coord].isel(sensor_args).values for coord in ['XC', 'YC']]

    # ASTE perturbation: mean and std across sensors
    xx_sensors = ds_ctrl.xx.isel(sensor_args).rename({'dim_0': 'sensor'})
    xx_mean = xx_sensors.mean('sensor')
    xx_std = xx_sensors.std('sensor')
    
    # plot timeseries with error envelope
    xx_mean.plot(c='b', ax=ax)
    ax.fill_between(
        ds_ctrl.time,
        xx_mean - xx_std,
        xx_mean + xx_std,
        color='b',
        alpha=0.3,
        label='ASTE ± std',
        edgecolor=None,
    )
    
    # JRA: mean and std across sensors
    jra_std_sensors = jra_std.sel(
        lon=xr.DataArray(cable_lons),
        lat=xr.DataArray(cable_lats),
        method="nearest"
    ).rename({'dim_0': 'sensor'})
    
    jra_std_sensors_mean = jra_std_sensors.mean('sensor')
    jra_std_sensors_std = jra_std_sensors.std('sensor')
    
    # plot timeseries with error envelope
    jra_std_sensors_mean.plot(c='r', ax=ax)
    ax.fill_between(
        jra_std_sensors_mean.time,
        jra_std_sensors_mean - jra_std_sensors_std,
        jra_std_sensors_mean + jra_std_sensors_std,
        color='r',
        alpha=0.3,
        label='JRA ± std',
        edgecolor=None,
    )
    
    # Aesthetics
    ax.set_xlim([ds_ctrl.time[0].values, ds_ctrl.time[-1].values])
    ax.set_ylabel('[Pa]', fontsize=20)
    ax.set_xlabel('Time', fontsize=20)
    ax.legend(fontsize=12)
    ax.grid()
    ax.set_title('Atm. Pressure Variability along cable', fontsize=20)
    ax.tick_params(axis='both', labelsize=12)
    
    return fig, ax



def load_jra(
        jra_dir = '/work/03901/atnguyen/jra55/',
        year = 2012,
        fld = 'pres',
    ):
    jra_fpath = jra_dir + f'jra55_{fld}_{year}'
    nx, ny = (320, 640)
    jra = utils.read_float32(jra_fpath)
    nt = int(len(jra)/nx/ny)
    jra = jra.reshape(nt, nx, ny)
    
    # create jra grid
    lon = np.arange(0, 0.5625 * ny, 0.5625)
    lon = (lon + 180) % 360 - 180
    lon = np.sort(lon)
    
    # Latitude array: construct cumulatively
    lat_increments = np.array([
        0.556914, 0.560202, 0.560946, 0.561227, 0.561363,
        0.561440, 0.561487, 0.561518, 0.561539, 0.561554,
        0.561566, 0.561575, 0.561582, 0.561587, 0.561592,
        *([0.561619268965519] * 289),
        0.561592, 0.561587, 0.561582, 0.561575, 0.561566,
        0.561554, 0.561539, 0.561518, 0.561487, 0.561440,
        0.561363, 0.561227, 0.560946, 0.560202, 0.556914
    ])
    
    lat = np.cumsum(np.insert(lat_increments, 0, -89.57009))
    
    time = pd.date_range(start=f"{year}-01-01", periods=nt, freq="3H")
    # Add coordinates to DataArray
    jra_3hr = xr.DataArray(
        jra,
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        name="jra"
    )
    return jra_3hr


