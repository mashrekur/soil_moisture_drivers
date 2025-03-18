#!/usr/bin/env python
# coding: utf-8

# In[19]:


import cdsapi
import time
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from IPython.display import HTML, display
import warnings


# In[2]:


c = cdsapi.Client()


# In[3]:


years = ['2002', '2003', '2004', '2005']


# In[4]:


for year in years:
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'total_precipitation', '2m_temperature',
            ],
            'year': year,
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                49, -124, 25,
                -66,
            ],
        },
        f'hourly_era5_conus_{year}.nc')
    
    print(f"Downloaded data for {year}")
    
    # Wait for 60 seconds before the next request
    if year != years[-1]:
        print("Waiting 60 seconds before next request...")
        time.sleep(60)

print("All downloads completed.")


# In[8]:


for year in years:
    filename = f'hourly_era5_conus_{year}.nc'
    
    # Open the dataset
    ds = xr.open_dataset(filename)
    
    print(f"\nContents of {filename}:")
    print(ds.info())
    
    print("\nVariable names:")
    for var_name in ds.variables:
        print(f"- {var_name}")
    
    print("\nDimensions:")
    for dim_name, dim_size in ds.dims.items():
        print(f"- {dim_name}: {dim_size}")
    
    # If 'expver' is a dimension, let's check its values
    if 'expver' in ds.dims:
        print("\nExpver values:")
        print(ds.expver.values)
    
    # Close the dataset
    ds.close()

print("Inspection completed.")


# In[10]:


# Function to plot histograms
def plot_histograms(data, variable, year, units):
    plt.figure(figsize=(10, 6))
    plt.hist(data.values.flatten(), bins=50, edgecolor='black')
    plt.title(f'Histogram of {variable} for {year}')
    plt.xlabel(f'{variable} ({units})')
    plt.ylabel('Frequency')
    plt.show()

# Function to calculate and plot monthly averages
def plot_monthly_averages(data, variable, year, units):
    monthly_mean = data.groupby('time.month').mean(['time', 'latitude', 'longitude'])
    
    plt.figure(figsize=(10, 6))
    monthly_mean.plot(marker='o')
    plt.title(f'Monthly Average {variable} for {year}')
    plt.xlabel('Month')
    plt.ylabel(f'{variable} ({units})')
    plt.xticks(range(1, 13))
    plt.grid(True)
    plt.show()

# Process each year
for year in years:
    filename = f'hourly_era5_conus_{year}.nc'
    
    # Open the dataset
    ds = xr.open_dataset(filename)
    
    # Process temperature
    temp = ds['t2m'] - 273.15  # Convert to Celsius
    temp_units = '°C'
    temp = temp.rename('Temperature')
    
    print(f"\nAnalyzing Temperature for {year}")
    plot_histograms(temp, 'Temperature', year, temp_units)
    plot_monthly_averages(temp, 'Temperature', year, temp_units)
    
    # Process precipitation
    precip = ds['tp'] * 1000  # Convert to mm
    precip_units = 'mm'
    precip = precip.rename('Precipitation')
    
    print(f"\nAnalyzing Precipitation for {year}")
    plot_histograms(precip, 'Precipitation', year, precip_units)
    plot_monthly_averages(precip, 'Precipitation', year, precip_units)
    
    # Close the dataset
    ds.close()

print("Analysis completed.")


# In[12]:


# Function to plot data on CONUS map
def plot_conus_map(data, title, cmap, units):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
    
    ax.set_extent([-125, -66, 25, 49], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)
    
    im = ax.pcolormesh(data.longitude, data.latitude, data, 
                       transform=ccrs.PlateCarree(), cmap=cmap)
    
    plt.colorbar(im, label=units, orientation='horizontal', pad=0.05)
    
    plt.title(title)
    plt.show()

# Process each year
for year in years:
    filename = f'hourly_era5_conus_{year}.nc'
    
    # Open the dataset
    ds = xr.open_dataset(filename)
    
    # Calculate monthly average for July (month 7)
    july_temp = ds['t2m'].sel(time=ds.time.dt.month == 7).mean('time') - 273.15
    july_precip = ds['tp'].sel(time=ds.time.dt.month == 7).mean('time') * 1000 * 24  # Convert to mm/day
    
    # Plot temperature
    plot_conus_map(july_temp, f'Average Temperature in July {year}', 'coolwarm', '°C')
    
    # Plot precipitation
    plot_conus_map(july_precip, f'Average Daily Precipitation in July {year}', 'Blues', 'mm/day')
    
    # Close the dataset
    ds.close()

print("Map creation completed.")


# In[22]:


# Suppress the specific MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=UserWarning, message="shading='flat'.*")

# Load data for 2002
ds = xr.open_dataset('hourly_era5_conus_2002.nc')

# Select the first week of July 2002
start_date = '2002-07-01'
end_date = '2002-07-07'
ds_week = ds.sel(time=slice(start_date, end_date))

# Function to create base map
def create_map(ax):
    ax.set_extent([-125, -66, 25, 49], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)

# Function to animate
def animate(i, var, title, cmap, vmin, vmax):
    ax.clear()
    create_map(ax)
    data = var.isel(time=i)
    im = ax.pcolormesh(data.longitude, data.latitude, data, 
                       transform=ccrs.PlateCarree(), cmap=cmap,
                       vmin=vmin, vmax=vmax, shading='auto')
    ax.set_title(f"{title} - {data.time.dt.strftime('%Y-%m-%d %H:%M').values}")
    return im,

# Function to create and display animation
def create_animation(var, title, cmap, vmin, vmax):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())

    anim = FuncAnimation(fig, animate, frames=len(var.time), 
                         fargs=(var, title, cmap, vmin, vmax),
                         interval=200, blit=True)

    plt.close(fig)  # Prevent display of static plot
    return HTML(anim.to_jshtml())

# Animate temperature
temp_data = ds_week['t2m'] - 273.15  # Convert to Celsius
temp_min, temp_max = temp_data.min().values, temp_data.max().values

print("Displaying Temperature Animation")
display(create_animation(temp_data, 'Temperature (°C)', 'coolwarm', temp_min, temp_max))

# Animate precipitation
precip_data = ds_week['tp'] * 1000  # Convert to mm
precip_min, precip_max = precip_data.min().values, precip_data.max().values

print("Displaying Precipitation Animation")
display(create_animation(precip_data, 'Precipitation (mm)', 'Blues', precip_min, precip_max))

print("Animation display completed.")

# Close the dataset
ds.close()


# In[21]:


# Suppress the specific MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=UserWarning, message="shading='flat'.*")


# Load all data
datasets = [xr.open_dataset(f'hourly_era5_conus_{year}.nc') for year in years]

# Combine datasets
combined_ds = xr.concat(datasets, dim='time')

# Function to create base map
def create_map(ax):
    ax.set_extent([-125, -66, 25, 49], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)

# Function to animate
def animate(i, var, title, cmap, vmin, vmax):
    ax.clear()
    create_map(ax)
    data = var.isel(time=i)
    im = ax.pcolormesh(data.longitude, data.latitude, data, 
                       transform=ccrs.PlateCarree(), cmap=cmap,
                       vmin=vmin, vmax=vmax, shading='auto')
    ax.set_title(f"{title} - {data.time.dt.strftime('%Y-%m-%d').values}")
    return im,

# Function to create and display animation
def create_animation(var, title, cmap, vmin, vmax):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())

    anim = FuncAnimation(fig, animate, frames=len(var.time), 
                         fargs=(var, title, cmap, vmin, vmax),
                         interval=100, blit=True)

    plt.close(fig)  # Prevent display of static plot
    return HTML(anim.to_jshtml())

# Animate temperature
temp_data = combined_ds['t2m'] - 273.15  # Convert to Celsius
temp_min, temp_max = temp_data.min().values, temp_data.max().values

print("Displaying Temperature Animation")
display(create_animation(temp_data, 'Daily Average Temperature (°C)', 'coolwarm', temp_min, temp_max))

# Animate precipitation
precip_data = combined_ds['tp'] * 1000 * 24  # Convert to mm/day
precip_min, precip_max = precip_data.min().values, precip_data.max().values

print("Displaying Precipitation Animation")
display(create_animation(precip_data, 'Daily Total Precipitation (mm/day)', 'Blues', precip_min, precip_max))

print("Animation display completed.")

# Close all datasets
for ds in datasets:
    ds.close()


# In[ ]:




