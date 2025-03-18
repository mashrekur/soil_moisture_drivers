#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from datetime import datetime
import pandas as pd


# In[3]:


def load_era5_data(file_path):
    return xr.open_dataset(file_path)

def load_station_info(file_path):
    return pd.read_csv(file_path)

def create_conus_map(ax):
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)
    ax.set_extent([-125, -66.5, 24, 49], crs=ccrs.PlateCarree())

def extract_station_data(era5_data, stations, variable):
    station_data = []
    for _, station in stations.iterrows():
        lat, lon = station['latitude'], station['longitude']
        data = era5_data[variable].sel(latitude=lat, longitude=lon, method='nearest')
        station_data.append(data)
    return xr.concat(station_data, dim='station')

def plot_era5_stations_on_conus(era5_data, stations, variable, time, title, cmap='viridis'):
    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.LambertConformal()})
    create_conus_map(ax)
    
    station_data = extract_station_data(era5_data, stations, variable)
    plot_data = station_data.sel(time=time)
    
    scatter = ax.scatter(stations['longitude'], stations['latitude'], c=plot_data, 
                         transform=ccrs.PlateCarree(), cmap=cmap, s=50)
    
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.08)
    cbar.set_label(variable)
    
    plt.title(f'{title} - {pd.to_datetime(time.values).strftime("%Y-%m-%d %H:%M")}')
    plt.show()

def main():
    era5_file = 'Data/era5_processed_data/era5_processed_data_2000.nc'
    stations_file = 'Data/complete_stations_info.csv'

    era5_data = load_era5_data(era5_file)
    stations = load_station_info(stations_file)

    print("Available variables:")
    print(list(era5_data.data_vars))

    print("\nDataset Info:")
    print(era5_data.info())

    print("\nStation Info:")
    print(stations.head())

    variable = 'temperature'
    time = era5_data.time[0]

    plot_era5_stations_on_conus(era5_data, stations, variable, time, 'ERA5 Temperature at Stations')

    station_data = extract_station_data(era5_data, stations, 'temperature')
    average_temp = station_data.mean(dim='time')

    fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.LambertConformal()})
    create_conus_map(ax)
    scatter = ax.scatter(stations['longitude'], stations['latitude'], c=average_temp, 
                         transform=ccrs.PlateCarree(), cmap='coolwarm', s=50)
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.08)
    cbar.set_label('Average Temperature (°C)')
    plt.title(f'ERA5 Average Temperature at Stations for {pd.to_datetime(era5_data.time[0].values).year}')
    plt.show()

    station_index = 0
    station_data = extract_station_data(era5_data, stations.iloc[[station_index]], 'temperature')
    plt.figure(figsize=(15, 5))
    plt.plot(station_data.time, station_data.squeeze())
    plt.title(f'Temperature Time Series for Station {stations.iloc[station_index]["station_name"]}')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.show()

    print("\nStation Information:")
    print(stations)
    print(f"\nTotal number of stations: {len(stations)}")


# In[5]:


if __name__ == "__main__":
    main()


# In[ ]:




