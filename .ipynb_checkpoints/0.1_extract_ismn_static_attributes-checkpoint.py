#!/usr/bin/env python
# coding: utf-8

# In[45]:


import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import csv
from scipy import stats
import numpy as np


# In[47]:


main_dir = "data/ismn_2000_2023/"
usda_ars_dir = "data/ismn_2000_2023/USDA-ARS"


# In[49]:


# Initialize an empty list to store the station information
station_info = []


# In[51]:


# Iterate through all the subfolders
for root, dirs, files in os.walk(main_dir):
    for dir in dirs:
        station_dir = os.path.join(root, dir)
        
        # Find the .stm files in the station directory
        stm_files = [file for file in os.listdir(station_dir) if file.endswith(".stm")]
        
        if stm_files:
            # Extract the data source and station name from the directory structure
            data_source = os.path.basename(root)
            station_name = dir
            
            # Read the header of the first .stm file to get the latitude and longitude
            with open(os.path.join(station_dir, stm_files[0]), "r") as f:
                header = f.readline().strip().split()
                latitude, longitude = float(header[3]), float(header[4])
            
            # Find the static variables CSV file
            static_vars_file = None
            for file in os.listdir(station_dir):
                if file.endswith("_static_variables.csv"):
                    static_vars_file = os.path.join(station_dir, file)
                    break
            
            if static_vars_file:
                # Read the static variables file
                df_static_vars = pd.read_csv(static_vars_file, sep=";")
                
                # Check if the 'quantity_name' column exists
                if 'quantity_name' in df_static_vars.columns:
                    # Extract relevant static variables
                    saturation = df_static_vars[df_static_vars["quantity_name"] == "saturation"]["value"].values if "saturation" in df_static_vars["quantity_name"].values else None
                    clay_fraction = df_static_vars[df_static_vars["quantity_name"] == "clay fraction"]["value"].values if "clay fraction" in df_static_vars["quantity_name"].values else None
                    organic_carbon = df_static_vars[df_static_vars["quantity_name"] == "organic carbon"]["value"].values if "organic carbon" in df_static_vars["quantity_name"].values else None
                    sand_fraction = df_static_vars[df_static_vars["quantity_name"] == "sand fraction"]["value"].values if "sand fraction" in df_static_vars["quantity_name"].values else None
                    silt_fraction = df_static_vars[df_static_vars["quantity_name"] == "silt fraction"]["value"].values if "silt fraction" in df_static_vars["quantity_name"].values else None
                    land_cover = df_static_vars[df_static_vars["quantity_name"] == "land cover classification"]["value"].values if "land cover classification" in df_static_vars["quantity_name"].values else None
                    climate = df_static_vars[df_static_vars["quantity_name"] == "climate classification"]["value"].values if "climate classification" in df_static_vars["quantity_name"].values else None
                else:
                    saturation = None
                    clay_fraction = None
                    organic_carbon = None
                    sand_fraction = None
                    silt_fraction = None
                    land_cover = None
                    climate = None
            else:
                saturation = None
                clay_fraction = None
                organic_carbon = None
                sand_fraction = None
                silt_fraction = None
                land_cover = None
                climate = None
            
            # Store the station information
            station_info.append({
                "data_source": data_source,
                "station_name": station_name,
                "saturation": saturation,
                "clay_fraction": clay_fraction,
                "organic_carbon": organic_carbon,
                "sand_fraction": sand_fraction,
                "silt_fraction": silt_fraction,
                "land_cover": land_cover,
                "climate": climate,
                "latitude": latitude,
                "longitude": longitude
            })

# Create a DataFrame from the station information
df_stations = pd.DataFrame(station_info)


# In[5]:


df_stations


# In[38]:


# Select only the necessary columns
stations_info = df_stations[['data_source', 'station_name', 'latitude', 'longitude']]

# Save to CSV
stations_info.to_csv('stations_info.csv', index=False)


# In[6]:


# Create a GeoDataFrame from the DataFrame
gdf_stations = gpd.GeoDataFrame(
    df_stations,
    geometry=gpd.points_from_xy(df_stations["longitude"], df_stations["latitude"])
)

# Load the CONUS shapefile
conus_shapefile = "Data/cb_2018_us_state_500k/cb_2018_us_state_500k.shp"
conus = gpd.read_file(conus_shapefile)

# Filter out Alaska, Hawaii, and territories
conus_states = conus[~conus['NAME'].isin(['Alaska', 'Hawaii', 'Puerto Rico', 'Guam', 'American Samoa', 'Commonwealth of the Northern Mariana Islands', 'United States Virgin Islands'])]

# Create a plot
fig, ax = plt.subplots(figsize=(15, 10))

# Plot the CONUS shapefile
conus_states.plot(ax=ax, alpha=0.5, edgecolor="black", color="white")

# Plot the stations
gdf_stations.plot(ax=ax, markersize=20, color="red")

# Set the extent of the plot to the bounds of the CONUS states
ax.set_xlim(conus_states.total_bounds[[0, 2]])
ax.set_ylim(conus_states.total_bounds[[1, 3]])

# Set the title and labels
ax.set_title("ISMN Stations in CONUS with Static Attributes")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Remove axis ticks
ax.set_xticks([])
ax.set_yticks([])

# Display the plot
plt.tight_layout()
plt.show()


# In[53]:


import pandas as pd
import numpy as np
import os

# Ensure the Data directory exists
if not os.path.exists('Data'):
    os.makedirs('Data')

# Save the DataFrame to a CSV file
df_stations.to_csv('Data/complete_stations_info.csv', index=False)
print("DataFrame saved to Data/complete_stations_info.csv")

# Convert DataFrame to structured numpy array
stations_array = df_stations.to_records(index=False)

# Save the array to a .npy file
np.save('Data/complete_stations_info.npy', stations_array)
print("Numpy array saved to Data/complete_stations_info.npy")

# Print some information about the saved array
print(f"\nArray shape: {stations_array.shape}")
print(f"Array data types:")
for name in stations_array.dtype.names:
    print(f"  {name}: {stations_array.dtype[name]}")

# Verify the saved data
loaded_array = np.load('Data/complete_stations_info.npy')
print(f"\nVerification: Loaded array has {len(loaded_array)} stations and {len(loaded_array.dtype.names)} attributes per station.")

print("\nProcess completed successfully.")


# In[55]:


df_stations


# In[ ]:




