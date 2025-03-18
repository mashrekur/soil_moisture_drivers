#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import requests
from tqdm import tqdm
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import numpy as np


# In[2]:


stations_path = 'stations_info.csv'
stations = pd.read_csv(stations_path)


# In[3]:


# Create a GeoDataFrame
geometry = [Point(xy) for xy in zip(stations['longitude'], stations['latitude'])]
stations_gdf = gpd.GeoDataFrame(stations, geometry=geometry, crs="EPSG:4326")

# Limit to first 5 stations
# stations_gdf = stations_gdf.head(5)


# In[4]:


# Create a session with retry logic
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))


# In[5]:


def get_soilgrids_data(lat, lon):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lat={lat}&lon={lon}"
    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        properties = data['properties']['layers']
        
        soil_data = {}
        for prop in properties:
            prop_name = prop['name']
            if prop_name in ['clay', 'sand', 'silt', 'soc', 'nitrogen', 'phh2o', 'awc',
                             'bdod', 'cec', 'cfvo', 'ocd', 'ocs', 'wv0010', 'wv0033', 'wv1500']:
                soil_data[prop_name] = prop['depths'][0]['values']['mean']
        
        return soil_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for lat={lat}, lon={lon}: {e}")
        return None


# In[6]:


# Fetch soil data for each station
soil_data_list = []
for idx, row in tqdm(stations_gdf.iterrows(), total=len(stations_gdf)):
    max_retries = 3
    for attempt in range(max_retries):
        soil_data = get_soilgrids_data(row.geometry.y, row.geometry.x)
        if soil_data:
            soil_data['station_name'] = row['station_name']
            soil_data_list.append(soil_data)
            break
        elif attempt < max_retries - 1:
            print(f"Retrying {row['station_name']} (Attempt {attempt + 2}/{max_retries})")
            time.sleep(5)  # Wait 5 seconds before retrying
    time.sleep(1)  # Increased delay between requests

# Create a DataFrame with the soil data
soil_df = pd.DataFrame(soil_data_list)

# Merge with the original stations data
result = pd.merge(stations, soil_df, on='station_name', how='left')

# Save the result
result.to_csv('stations_with_soilgrids_data.csv', index=False)
print("Process completed. Results saved to 'stations_with_soilgrids_data.csv'")

# Print summary of missing data
missing_data = result[soil_df.columns].isnull().sum()
print("\nSummary of missing data:")
print(missing_data)
print(f"\nTotal stations processed: {len(result)}")
print(f"Stations with complete soil data: {len(result) - missing_data.max()}")


# In[9]:


# Load the results
result = pd.read_csv('stations_with_soilgrids_data.csv')

# Load a US states shapefile 
us_states = gpd.read_file('Data/cb_2018_us_state_500k/cb_2018_us_state_500k.shp')

# Convert result to GeoDataFrame
gdf = gpd.GeoDataFrame(
    result, geometry=gpd.points_from_xy(result.longitude, result.latitude), crs="EPSG:4326"
)

# List of parameters to visualize
parameters = ['clay', 'sand', 'silt', 'soc', 'nitrogen', 'phh2o',
              'bdod', 'cec', 'cfvo', 'ocd', 'ocs', 'wv0010', 'wv0033', 'wv1500']

# Create a plot for each parameter
for param in parameters:
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot US states
    us_states.to_crs(gdf.crs).plot(ax=ax, color='lightgrey', edgecolor='black')
    
    # Plot stations, coloring by parameter
    scatter = gdf.plot(ax=ax, column=param, cmap='viridis', 
                       legend=True, markersize=50, edgecolor='black')
    
    # Improve the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(scatter.collections[0], cax=cax)
    
    # Set title and labels
    plt.title(f'{param.upper()} Distribution Across ISMN Stations', fontsize=16)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Remove axis ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'Figures/{param}_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

print("Visualization completed. Check the current directory for the output images.")


# In[14]:


# Load the results
result = pd.read_csv('stations_with_soilgrids_data.csv')

# Function to plot data on CONUS map
def plot_conus_map(data, column, title, cmap, units):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())
    
    ax.set_extent([-125, -66, 25, 49], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES)
    
    # Plot the data points
    scatter = ax.scatter(data['longitude'], data['latitude'], c=data[column], 
                         transform=ccrs.PlateCarree(), cmap=cmap, 
                         edgecolor='black', linewidth=0.5, s=50)
    
    plt.colorbar(scatter, label=units, orientation='horizontal', pad=0.05)
    
    plt.title(title)
    return fig

# List of parameters to visualize
# List of parameters to visualize
parameters = {
    'clay': ('Clay Content', 'YlOrBr', 'g/kg'),
    'sand': ('Sand Content', 'YlOrRd', 'g/kg'),
    'silt': ('Silt Content', 'YlGnBu', 'g/kg'),
    'soc': ('Soil Organic Carbon', 'Greens', 'g/kg'),
    'nitrogen': ('Nitrogen', 'Purples', 'cg/kg'),
    'phh2o': ('pH in H2O', 'RdYlBu', 'pH'),
    'bdod': ('Bulk Density', 'Greys', 'cg/cm3'),
    'cec': ('Cation Exchange Capacity', 'OrRd', 'mmol(c)/kg'),
    'cfvo': ('Coarse Fragments', 'BuPu', 'cm3/dm3'),
    'ocd': ('Organic Carbon Density', 'GnBu', 'kg/dm3'),
    'ocs': ('Organic Carbon Stocks', 'YlGn', 'kg/m2'),
    'wv0010': ('Water Content at pF 0', 'Blues', 'cm3/cm3'), #saturated water content
    'wv0033': ('Water Content at pF 2.5', 'Blues', 'cm3/cm3'), #field capacity
    'wv1500': ('Water Content at pF 4.2', 'Blues', 'cm3/cm3') #wilting point
}


# Create a plot for each parameter
for param, (title, cmap, units) in parameters.items():
    if param in result.columns:
        fig = plot_conus_map(result, param, f'{title} Distribution Across ISMN Stations', cmap, units)
        fig.savefig(f'Figures/{param}_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print(f"Warning: {param} not found in the data.")



# In[ ]:




