#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
import numpy as np
from typing import Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from mpl_toolkits.basemap import Basemap
import io
from PIL import Image


# In[2]:


# Load the stations info
stations_info = pd.read_csv('stations_info.csv')


# In[3]:


# Counter for missing elevations
missing_elevations = 0


# In[4]:


def get_elevation(lat: float, lon: float) -> float:
    global missing_elevations
    url = f"https://epqs.nationalmap.gov/v1/json?x={lon}&y={lat}&units=Meters&output=json" #elevation point query service
    response = requests.get(url)
    elevation_str = response.json()['value']
    if elevation_str == 'None':
        missing_elevations += 1
        return None
    return float(elevation_str)

def calculate_slope(center: Tuple[float, float], distance: float = 0.001) -> float:
    lat, lon = center
    elevations = [
        get_elevation(lat + distance, lon),  # North
        get_elevation(lat - distance, lon),  # South
        get_elevation(lat, lon + distance),  # East
        get_elevation(lat, lon - distance),  # West
    ]
    center_elevation = get_elevation(lat, lon)
    
    if None in elevations or center_elevation is None:
        return None
    
    # Calculate slope using central difference
    dy = (elevations[0] - elevations[1]) / (2 * distance * 111000)  # North-South gradient
    dx = (elevations[2] - elevations[3]) / (2 * distance * 111000 * np.cos(np.radians(lat)))  # East-West gradient
    
    # Calculate the steepest slope
    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * 180 / np.pi
    
    # Determine if it's uphill or downhill
    if center_elevation < np.mean(elevations):
        slope = -slope
    
    return slope

def get_slope_for_station(row):
    try:
        return calculate_slope((row['latitude'], row['longitude']))
    except Exception as e:
        print(f"Error calculating slope for {row['station_name']}: {e}")
        return None


# In[5]:


# Calculate slopes for all stations with progress bar
tqdm.pandas(desc="Calculating slopes")
stations_info['slope'] = stations_info.progress_apply(get_slope_for_station, axis=1)

# Display the results
print(stations_info)

# Print summary of missing elevations
total_queries = len(stations_info) * 5  # 5 elevation queries per station
print(f"\nMissing Elevations Summary:")
print(f"Total elevation queries: {total_queries}")
print(f"Missing elevations: {missing_elevations}")
print(f"Percentage of missing elevations: {(missing_elevations / total_queries) * 100:.2f}%")

# Count and display the number of stations with None slope
stations_with_none_slope = stations_info['slope'].isna().sum()
print(f"\nStations with None slope: {stations_with_none_slope}")
print(f"Percentage of stations with None slope: {(stations_with_none_slope / len(stations_info)) * 100:.2f}%")

# Optionally, save the results to a new CSV
stations_info.to_csv('stations_with_slopes_updn.csv', index=False)


# In[11]:


# Load the data
df = pd.read_csv('stations_with_slopes_updn.csv')

# Remove rows with missing slopes for visualization
df_clean = df.dropna(subset=['slope'])

# 2D Colormap
plt.figure(figsize=(15, 10))
m = Basemap(llcrnrlon=-125, llcrnrlat=24, urcrnrlon=-66, urcrnrlat=50,
            projection='lcc', lat_1=33, lat_2=45, lon_0=-95)

m.drawcoastlines()
m.drawcountries()
m.drawstates()

x, y = m(df_clean['longitude'].values, df_clean['latitude'].values)
scatter = m.scatter(x, y, c=df_clean['slope'], cmap='RdYlBu', s=20, alpha=0.7, vmin=-max(abs(df_clean['slope'])), vmax=max(abs(df_clean['slope'])))

plt.colorbar(scatter, label='Slope (degrees)')
plt.title('Topographic Slopes at ISMN Stations')
plt.savefig('2D_slope_map.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# In[12]:


print(df_clean['slope'].describe())
plt.figure(figsize=(10, 6))
plt.hist(df_clean['slope'], bins=50)
plt.title('Distribution of Slope Values')
plt.xlabel('Slope (degrees)')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# In[8]:


# 3D Contour Map with basemap
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

# Create a grid for interpolation
xi = np.linspace(df_clean['longitude'].min(), df_clean['longitude'].max(), 100)
yi = np.linspace(df_clean['latitude'].min(), df_clean['latitude'].max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate
zi = griddata((df_clean['longitude'], df_clean['latitude']), df_clean['slope'], (xi, yi), method='cubic')

# Plot the surface
surf = ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.8)

# Add basemap
m = Basemap(llcrnrlon=-125, llcrnrlat=24, urcrnrlon=-66, urcrnrlat=50,
            projection='lcc', lat_1=33, lat_2=45, lon_0=-95)

# Capture basemap as an image
img = io.BytesIO()
plt.figure(figsize=(15, 10))
m.drawcoastlines()
m.drawcountries()
m.drawstates()
plt.savefig(img, format='png')
plt.close()

# Load the image and display it on the 3D plot
img = Image.open(img)
img = np.array(img)

# Normalize the image array to 0-1 range
img_normalized = img.astype(float) / 255.0

# Plot the basemap on the 3D surface
ax.plot_surface(xi, yi, np.zeros_like(zi), rstride=5, cstride=5, facecolors=img_normalized)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Slope (degrees)')
ax.set_title('3D Contour Map of Topographic Slopes')

plt.colorbar(surf, label='Slope (degrees)')
# Save and display
plt.savefig('3D_slope_map.png', dpi=300, bbox_inches='tight')
plt.show()


# In[9]:


# Print summary of missing data
total_stations = len(df)
missing_slopes = df['slope'].isna().sum()
print(f"Total stations: {total_stations}")
print(f"Stations with missing slopes: {missing_slopes}")
print(f"Percentage of stations with missing slopes: {missing_slopes/total_stations*100:.2f}%")


# In[ ]:




