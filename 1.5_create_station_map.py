#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Previous imports plus:
from matplotlib.patches import Rectangle
from textwrap import wrap

# [Previous code until figure creation]

# Create figure with gridspec for layout control
fig = plt.figure(figsize=(20, 10), dpi=600)  # Wider figure to accommodate list
gs = fig.add_gridspec(1, 2, width_ratios=[0.3, 0.7])  # 30% for list, 70% for map

# Create axis for station list
ax_list = fig.add_subplot(gs[0])
ax_map = fig.add_subplot(gs[1], projection=ccrs.AlbersEqualArea(
    central_longitude=-96,
    central_latitude=38,
    standard_parallels=(29.5, 45.5)
))

# Get sorted station names
station_names = sorted(static_ds.station_name.values)
n_stations = len(station_names)
stations_per_column = n_stations // 3 + (1 if n_stations % 3 != 0 else 0)

# Create station list text
stations_text = []
for i in range(0, n_stations, stations_per_column):
    column = station_names[i:i + stations_per_column]
    stations_text.append('\n'.join(column))

# Plot station list
ax_list.text(0.1, 0.98, 'ISMN Stations Used:', 
             fontsize=12, fontweight='bold', 
             verticalalignment='top')

x_positions = [0.1, 0.35, 0.7]
for i, text in enumerate(stations_text):
    ax_list.text(x_positions[i], 0.95, text,
                fontsize=6,
                verticalalignment='top',
                fontfamily='monospace')

# Remove axes from station list
ax_list.axis('off')

# Add background color to station list panel
ax_list.add_patch(Rectangle((0, 0), 1, 1, facecolor='white', alpha=0.8))

# [Previous mapping code but using ax_map instead of ax]
ax_map.set_extent([-125, -66.5, 24, 49], crs=ccrs.PlateCarree())

# Add features with higher resolution
ax_map.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', alpha=0.5)
ax_map.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='lightblue', alpha=0.5)

# Add state boundaries
states = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m',
    facecolor='none'
)
ax_map.add_feature(states, edgecolor='gray', linewidth=0.5)

# Add country borders
countries = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_boundary_lines_land',
    scale='10m',
    facecolor='none'
)
ax_map.add_feature(countries, edgecolor='black', linewidth=1.0)

# Add coastlines
ax_map.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)

# Plot stations
scatter = ax_map.scatter(stations_used['longitude'], 
                     stations_used['latitude'],
                     c='red',
                     s=60,
                     alpha=0.8,
                     transform=ccrs.PlateCarree(),
                     edgecolor='white',
                     linewidth=0.5,
                     label=f'ISMN Stations (n={len(stations_used)})\nPost-filtering Time Period: 01/01/2015 - 12/31/2019')

# Add state abbreviations
for state, coords in state_coords.items():
    ax_map.text(coords[0], coords[1], state,
             transform=ccrs.PlateCarree(),
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=8,
             fontweight='bold',
             color='black')

# Add gridlines to map
gl = ax_map.gridlines(draw_labels=True, 
                    linewidth=0.5, 
                    color='gray', 
                    alpha=0.3, 
                    linestyle=':')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

# Add title
plt.suptitle('ISMN Stations Used in This Study', 
             fontsize=16, 
             y=0.95,
             fontweight='bold')

# Add legend to map
legend = ax_map.legend(loc='lower right', 
                    frameon=True,
                    framealpha=0.9,
                    facecolor='white',
                    edgecolor='black',
                    fontsize=10)

# Adjust layout
plt.tight_layout()

# Save figure
os.makedirs('Figures', exist_ok=True)
plt.savefig('Figures/ismn_stations_map.png', 
            dpi=600, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            pad_inches=0.1)

plt.close()


# In[17]:


import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import os
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

# Get the latest clipped data directory
base_dir = 'Data/analysis_output'
clipped_dirs = [d for d in os.listdir(base_dir) if d.startswith('clipped_data_')]
latest_dir = sorted(clipped_dirs)[-1]
data_dir = os.path.join(base_dir, latest_dir)

# Load the static data from clipped dataset
static_ds = xr.open_dataset(os.path.join(data_dir, 'aligned_static_data.nc'))
stations_used = pd.DataFrame({
    'latitude': static_ds.latitude.values,
    'longitude': static_ds.longitude.values
})

# State abbreviations and their centroids
state_coords = {
    'AL': (-86.79113, 32.806671), 'AZ': (-111.431221, 34.048928),
    'AR': (-92.373123, 34.969704), 'CA': (-119.681564, 36.116203),
    'CO': (-105.311104, 39.059811), 'CT': (-72.755371, 41.597782),
    'DE': (-75.507141, 39.318523), 'FL': (-81.686783, 27.664827),
    'GA': (-83.643074, 32.157435), 'ID': (-114.478828, 44.240459),
    'IL': (-88.986137, 40.349457), 'IN': (-86.258278, 39.849426),
    'IA': (-93.210526, 42.011539), 'KS': (-96.726486, 38.526600),
    'KY': (-84.670067, 37.668140), 'LA': (-91.867805, 31.169546),
    'ME': (-69.381927, 45.353174), 'MD': (-76.802101, 39.063946),
    'MA': (-71.530106, 42.230171), 'MI': (-84.536095, 43.326618),
    'MN': (-93.900192, 45.694454), 'MS': (-89.678696, 32.741646),
    'MO': (-92.288368, 38.456085), 'MT': (-110.454353, 46.921925),
    'NE': (-98.268082, 41.125370), 'NV': (-117.055374, 38.313515),
    'NH': (-71.563896, 43.452492), 'NJ': (-74.521011, 40.298904),
    'NM': (-106.248482, 34.840515), 'NY': (-74.948051, 42.165726),
    'NC': (-79.806419, 35.630066), 'ND': (-99.784012, 47.528912),
    'OH': (-82.764915, 40.388783), 'OK': (-96.928917, 35.565342),
    'OR': (-122.070938, 44.572021), 'PA': (-77.209755, 40.590752),
    'RI': (-71.477429, 41.680893), 'SC': (-80.945007, 33.856892),
    'SD': (-99.438828, 44.299782), 'TN': (-86.692345, 35.747845),
    'TX': (-97.563461, 31.054487), 'UT': (-111.862434, 40.150032),
    'VT': (-72.710686, 44.045876), 'VA': (-78.169968, 37.769337),
    'WA': (-121.490494, 47.400902), 'WV': (-80.954453, 38.491226),
    'WI': (-89.616508, 44.268543), 'WY': (-107.290284, 42.755966)
}

# Create figure
plt.figure(figsize=(15, 10), dpi=600)

# Create map projection
ax = plt.axes(projection=ccrs.AlbersEqualArea(
    central_longitude=-96,
    central_latitude=38,
    standard_parallels=(29.5, 45.5)
))

# Set map bounds
ax.set_extent([-125, -66.5, 24, 49], crs=ccrs.PlateCarree())

# Add features with higher resolution
ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray', alpha=0.5)
ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='lightblue', alpha=0.5)

# Add state boundaries
states = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='10m',
    facecolor='none'
)
ax.add_feature(states, edgecolor='gray', linewidth=0.5)

# Add country borders with higher resolution
countries = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_0_boundary_lines_land',
    scale='10m',
    facecolor='none'
)
ax.add_feature(countries, edgecolor='black', linewidth=1.0)

# Add coastlines
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)

# Plot stations
scatter = ax.scatter(stations_used['longitude'], 
                    stations_used['latitude'],
                    c='red',
                    s=60,
                    alpha=0.8,
                    transform=ccrs.PlateCarree(),
                    edgecolor='white',
                    linewidth=0.5,
                    label=f'ISMN Stations (n={len(stations_used)})\nPost-filtering Time Period: 01/01/2015 - 12/31/2019')

# Add state abbreviations
for state, coords in state_coords.items():
    ax.text(coords[0], coords[1], state,
            transform=ccrs.PlateCarree(),
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=8,
            fontweight='bold',
            color='black')

# Add gridlines
gl = ax.gridlines(draw_labels=True, 
                 linewidth=0.5, 
                 color='gray', 
                 alpha=0.3, 
                 linestyle=':')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

# Add title
plt.title('ISMN Stations Used in This Study', 
          fontsize=16, 
          pad=20,
          fontweight='bold')

# Add legend
legend = ax.legend(loc='lower right', 
                  frameon=True,
                  framealpha=0.9,
                  facecolor='white',
                  edgecolor='black',
                  fontsize=10)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('Figures/ismn_filtered_stations_map.png', 
            dpi=600, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none',
            pad_inches=0.1)

plt.close()


# In[11]:


#visualize static and dynamic maps

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
import os
import math

def get_latest_data_paths():
    """Get the paths to the most recent clipped data"""
    base_dir = 'Data/analysis_output'
    clipped_dirs = [d for d in os.listdir(base_dir) if d.startswith('clipped_data_')]
    
    if not clipped_dirs:
        raise ValueError("No clipped data directories found!")
    
    # Get most recent directory
    latest_dir = sorted(clipped_dirs)[-1]
    data_dir = os.path.join(base_dir, latest_dir)
    
    print(f"Using data from: {data_dir}")
    
    return {
        'static_data_path': os.path.join(data_dir, 'aligned_static_data.nc'),
        'dynamic_data_path': os.path.join(data_dir, 'aligned_dynamic_data.nc'),
        'target_data_path': os.path.join(data_dir, 'clipped_target_data.nc')
    }

def create_conus_maps():
    # Load data
    data_paths = get_latest_data_paths()
    stations_info_path = os.path.join('Data', 'complete_stations_info.csv')
    
    print("Loading datasets...")
    static_ds = xr.open_dataset(data_paths['static_data_path'])
    dynamic_ds = xr.open_dataset(data_paths['dynamic_data_path'])
    stations_info = pd.read_csv(stations_info_path)
    
    # Get the station names from the static dataset
    static_stations = sorted(static_ds.station_name.values)
    stations_info = stations_info[stations_info['station_name'].isin(static_stations)]
    stations_info = stations_info.drop_duplicates('station_name', keep='first')
    stations_info = stations_info.sort_values('station_name').reset_index(drop=True)
    
    # Create flat list of all variables with their metadata
    dynamic_cols = ['temperature', 'precipitation', 'solar_radiation']
    variables = [
        # Static Physical
        ('slope', 'Slope', 'degrees'),
        ('cfvo', 'Coarse Fragments', 'cm³/dm³'),
        ('bdod', 'Bulk Density', 'kg/dm³'),
        ('silt_fraction_2', 'Silt Fraction (30-60cm)', '%'),
        ('clay_fraction_2', 'Clay Fraction (30-60cm)', '%'),
        ('clay_fraction_1', 'Clay Fraction (0-30cm)', '%'),
        ('silt_fraction_1', 'Silt Fraction (0-30cm)', '%'),
        ('saturation_2', 'Saturation (30-60cm)', '%'),
        ('saturation_1', 'Saturation (0-30cm)', '%'),
        ('sand_fraction_2', 'Sand Fraction (30-60cm)', '%'),
        ('sand_fraction_1', 'Sand Fraction (0-30cm)', '%'),
        # Static Chemical
        ('ocs', 'Organic Carbon Stock', 'kg/m²'),
        ('cec', 'CEC', 'mmol(c)/kg'),
        ('nitrogen', 'Nitrogen', 'g/kg'),
        ('ocd', 'Organic Carbon Density', 'kg/dm³'),
        ('phh2o', 'pH in H2O', 'pH'),
        ('soc', 'Soil Organic Carbon', 'g/kg'),
        ('organic_carbon_2', 'Organic Carbon (30-60cm)', 'g/kg'),
        ('organic_carbon_1', 'Organic Carbon (0-30cm)', 'g/kg'),
        # Static Hydraulic
        ('wv0033', 'Water Content at pF 2.5', 'cm³/cm³'),
        ('wv1500', 'Water Content at pF 4.2', 'cm³/cm³'),
        ('wv0010', 'Water Content at pF 2.0', 'cm³/cm³'),
        # Environmental
        ('landcover_2020_encoded', 'Land Cover', ''),
        ('climate_2_encoded', 'Secondary Climate', ''),
        ('climate_1_encoded', 'Primary Climate', ''),
        # Dynamic Weather
        ('temperature', 'Temperature', '°C'),
        ('precipitation', 'Precipitation', 'mm/h'),
        ('solar_radiation', 'Solar Radiation', 'W/m²')
    ]
    
    # Calculate grid dimensions
    n_vars = len(variables)
    n_cols = 4  # We want 6 columns
    n_rows = math.ceil(n_vars / n_cols)
    
    print(f"\nGrid dimensions: {n_rows} rows x {n_cols} columns")
    
    # Create figure with adjusted spacing
    fig = plt.figure(figsize=(24, math.ceil(24 * n_rows/n_cols)))
    gs = GridSpec(n_rows, n_cols, figure=fig, 
                 left=0.05, right=0.95,
                 bottom=0.05, top=0.5,
                 wspace=0.1, hspace=0.1)
    
    # Set up base map properties
    proj = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=38)
    
    # Plot variables
    for idx, (var_name, title, units) in enumerate(variables):
        row = idx // n_cols
        col = idx % n_cols
        
        ax = fig.add_subplot(gs[row, col], projection=proj)
        
        # Set extent for CONUS
        ax.set_extent([-125, -66.5, 24, 49], crs=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.STATES.with_scale('50m'), edgecolor='gray', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
        
        # Get data for variable
        if var_name in dynamic_cols:
            time_slice = dynamic_ds.time[1000]
            data = dynamic_ds[var_name].sel(time=time_slice).values
        else:
            data = static_ds[var_name].values
        
        # Create scatter plot
        scatter = ax.scatter(stations_info['longitude'], 
                           stations_info['latitude'],
                           c=data, 
                           transform=ccrs.PlateCarree(),
                           cmap='viridis',
                           s=30,
                           alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(f'{units}', fontsize=8)
        
        # Add title
        ax.set_title(f'{title}', fontsize=14, pad=5)
    
    plt.tight_layout()
    
    # Save figure
    print("\nSaving figure...")
    plt.savefig('Figures/conus_variable_maps_grid.png', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white')
    plt.close()

# Create maps
create_conus_maps()


# In[ ]:




