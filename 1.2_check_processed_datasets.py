#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from IPython.display import display
import pickle

def preprocess_soil_moisture(da):
    # Convert to pandas Series for easier manipulation
    series = da.to_series()
    
    # Set negative values to zero
    series[series < 0] = 0
    
    # Identify gaps
    gaps = series.isna()
    
    # Fill gaps with mean of previous and next valid values
    for i in range(len(series)):
        if gaps[i]:
            prev_valid = series.iloc[:i].last_valid_index()
            next_valid = series.iloc[i:].first_valid_index()
            
            if prev_valid is not None and next_valid is not None:
                series.iloc[i] = (series.loc[prev_valid] + series.loc[next_valid]) / 2
            elif prev_valid is not None:
                series.iloc[i] = series.loc[prev_valid]
            elif next_valid is not None:
                series.iloc[i] = series.loc[next_valid]
    
    return xr.DataArray(series, coords=da.coords, dims=da.dims)

def analyze_and_visualize_data(target_file, static_file, dynamic_file, output_dir, resume=True):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_file = os.path.join(output_dir, 'preprocessing_checkpoint.pkl')
    
    if resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        target_ds = checkpoint_data['target_ds']
        preprocessed_stations = checkpoint_data['preprocessed_stations']
        print(f"Resumed preprocessing. {len(preprocessed_stations)} stations already processed.")
    else:
        target_ds = xr.open_dataset(target_file)
        preprocessed_stations = []

    # 1. Preprocess and analyze soil moisture data
    for station in target_ds.data_vars:
        if station not in preprocessed_stations:
            original_data = target_ds[station]
            preprocessed_data = preprocess_soil_moisture(original_data)
            target_ds[station] = preprocessed_data
            
            if not np.array_equal(original_data, preprocessed_data):
                preprocessed_stations.append(station)
            
            # Save checkpoint after each station
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({'target_ds': target_ds, 'preprocessed_stations': preprocessed_stations}, f)
    
    print(f"Stations that required preprocessing: {len(preprocessed_stations)}")
    
    # Visualize preprocessed stations
    if preprocessed_stations:
        fig, axs = plt.subplots(min(5, len(preprocessed_stations)), 1, figsize=(15, 5*min(5, len(preprocessed_stations))), sharex=True)
        if len(preprocessed_stations) == 1:
            axs = [axs]
        for i, station in enumerate(preprocessed_stations[:5]):
            data = target_ds[station]
            axs[i].plot(data.index, data.values)
            axs[i].set_title(f"{station} (preprocessed)")
            axs[i].set_ylabel('Soil Moisture')
            axs[i].set_xlabel('Time')
        plt.tight_layout()
        display(fig)
        plt.savefig(os.path.join(output_dir, 'preprocessed_stations.png'))
        plt.close()
    
    # Load static and dynamic datasets
    static_ds = xr.open_dataset(static_file)
    dynamic_ds = xr.open_dataset(dynamic_file)

    # Print dataset structures
    for ds_name, ds in [('Static', static_ds), ('Dynamic', dynamic_ds), ('Target', target_ds)]:
        print(f"\n{ds_name} Dataset Structure:")
        print(ds)
        print(f"\n{ds_name} Dataset Data Variables:")
        for var in ds.data_vars:
            print(f"{var}: {ds[var].shape}")

    # Align datasets
    processed_stations = list(target_ds.data_vars)
    
    # Filter static data
    common_stations_static = list(set(processed_stations) & set(static_ds.index.values))
    filtered_static_ds = static_ds.sel(index=common_stations_static)

    # Filter dynamic data
    if 'dim_0' in dynamic_ds.dims:
        station_to_dim0 = dict(zip(dynamic_ds.station.values, dynamic_ds.dim_0.values))
        common_stations_dynamic = list(set(processed_stations) & set(dynamic_ds.station.values))
        common_dim0_indices = [station_to_dim0[station] for station in common_stations_dynamic]
        filtered_dynamic_ds = dynamic_ds.sel(dim_0=common_dim0_indices)
    else:
        print("Warning: 'dim_0' dimension not found in dynamic dataset. Using original dataset.")
        filtered_dynamic_ds = dynamic_ds

    # Save filtered datasets
    for ds_name, ds in [('static', filtered_static_ds), ('dynamic', filtered_dynamic_ds), ('target', target_ds)]:
        try:
            ds.to_netcdf(os.path.join(output_dir, f'filtered_{ds_name}_data.nc'))
            print(f"Filtered {ds_name} data saved successfully")
        except Exception as e:
            print(f"Error saving filtered {ds_name} data: {str(e)}")

    # Analyze data alignment
    all_stations = set(processed_stations)
    static_stations = set(static_ds.index.values)
    dynamic_stations = set(dynamic_ds.station.values)

    print("\nData Alignment Analysis:")
    print(f"Total processed target stations: {len(all_stations)}")
    print(f"Stations in static data: {len(static_stations)}")
    print(f"Stations in dynamic data: {len(dynamic_stations)}")
    print(f"Stations in all three datasets: {len(all_stations & static_stations & dynamic_stations)}")
    print(f"Stations missing from static data: {len(all_stations - static_stations)}")
    print(f"Stations missing from dynamic data: {len(all_stations - dynamic_stations)}")

# Usage
target_file = 'Data/combined_ismn_data/target_data.nc'
static_file = 'Data/combined_ismn_data/static_data.nc'
dynamic_file = 'Data/combined_ismn_data/dynamic_data.nc'
output_dir = 'Data/analysis_output'

analyze_and_visualize_data(target_file, static_file, dynamic_file, output_dir, resume=True)


# In[5]:


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from IPython.display import display

def analyze_dataset_structure(ds, name):
    print(f"\n{name} Dataset Structure:")
    print(ds)
    print(f"\n{name} Dataset Dimensions:")
    for dim in ds.dims:
        print(f"{dim}: {ds.dims[dim]}")
    print(f"\n{name} Dataset Coordinates:")
    for coord in ds.coords:
        print(f"{coord}: {ds.coords[coord].shape}")
    print(f"\n{name} Dataset Data Variables:")
    for var in ds.data_vars:
        print(f"{var}: {ds[var].shape}")

def analyze_and_visualize_data(target_file, static_file, dynamic_file, output_dir):
    # Load datasets
    target_ds = xr.open_dataset(target_file)
    static_ds = xr.open_dataset(static_file)
    dynamic_ds = xr.open_dataset(dynamic_file)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Analyze structure of each dataset
    analyze_dataset_structure(target_ds, "Target")
    analyze_dataset_structure(static_ds, "Static")
    analyze_dataset_structure(dynamic_ds, "Dynamic")

    # Get station names or identifiers from each dataset
    target_stations = set(target_ds.data_vars)
    static_stations = set(static_ds.index.values)
    dynamic_stations = set(dynamic_ds.station.values)

    print("\nStation counts:")
    print(f"Target stations: {len(target_stations)}")
    print(f"Static stations: {len(static_stations)}")
    print(f"Dynamic stations: {len(dynamic_stations)}")

    # Find common stations
    common_stations = target_stations & static_stations & dynamic_stations
    print(f"\nCommon stations across all datasets: {len(common_stations)}")

    if len(common_stations) == 0:
        print("No common stations found. Investigating discrepancies...")
        print(f"Stations in target but not in static: {len(target_stations - static_stations)}")
        print(f"Stations in target but not in dynamic: {len(target_stations - dynamic_stations)}")
        print(f"Stations in static but not in target: {len(static_stations - target_stations)}")
        print(f"Stations in dynamic but not in target: {len(dynamic_stations - target_stations)}")

        # Print a few example station names from each dataset
        print("\nExample station names:")
        print("Target:", list(target_stations)[:5])
        print("Static:", list(static_stations)[:5])
        print("Dynamic:", list(dynamic_stations)[:5])

    else:
        # If common stations exist, proceed with filtering and analysis
        filtered_target_ds = target_ds[list(common_stations)]
        filtered_static_ds = static_ds.sel(index=[i for i in static_ds.index.values if i in common_stations])
        filtered_dynamic_ds = dynamic_ds.sel(station=list(common_stations))

        # Save filtered datasets
        for ds_name, ds in [("static", filtered_static_ds), ("dynamic", filtered_dynamic_ds), ("target", filtered_target_ds)]:
            try:
                ds.to_netcdf(os.path.join(output_dir, f'filtered_{ds_name}_data.nc'))
                print(f"Filtered {ds_name} data saved successfully")
            except Exception as e:
                print(f"Error saving filtered {ds_name} data: {str(e)}")

        # Analyze data alignment
        print("\nData Alignment Analysis:")
        print(f"Total common stations: {len(common_stations)}")
        print(f"Stations in filtered static data: {len(filtered_static_ds.index)}")
        print(f"Stations in filtered dynamic data: {len(filtered_dynamic_ds.station)}")
        print(f"Stations in filtered target data: {len(filtered_target_ds.data_vars)}")

# Usage
target_file = 'Data/combined_ismn_data/target_data.nc'
static_file = 'Data/combined_ismn_data/static_data.nc'
dynamic_file = 'Data/combined_ismn_data/dynamic_data.nc'
output_dir = 'Data/analysis_output'

analyze_and_visualize_data(target_file, static_file, dynamic_file, output_dir)


# In[11]:


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from IPython.display import display

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File does not exist: {file_path}")
        return False
    return True

def safe_open_dataset(file_path):
    if not check_file_exists(file_path):
        return None
    
    try:
        with xr.open_dataset(file_path) as ds:
            # Load the dataset into memory
            ds.load()
        return ds
    except Exception as e:
        print(f"Error opening {file_path}: {str(e)}")
        print(f"File size: {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")
        return None

def analyze_dataset(ds, name):
    if ds is None:
        print(f"{name} dataset could not be loaded.")
        return

    print(f"\n{name} Dataset Structure:")
    print(ds)
    print(f"\n{name} Dataset Dimensions:")
    for dim in ds.dims:
        print(f"{dim}: {ds.dims[dim]}")
    print(f"\n{name} Dataset Coordinates:")
    for coord in ds.coords:
        print(f"{coord}: {ds.coords[coord].shape}")
    print(f"\n{name} Dataset Data Variables:")
    for var in ds.data_vars:
        print(f"{var}: {ds[var].shape}")

def analyze_and_visualize_data(target_file, static_file, dynamic_file, output_dir):
    # Check if files exist
    for file_path in [target_file, static_file, dynamic_file]:
        check_file_exists(file_path)

    # Load datasets
    print("\nLoading target dataset...")
    target_ds = safe_open_dataset(target_file)
    
    print("\nLoading static dataset...")
    static_ds = safe_open_dataset(static_file)
    
    print("\nLoading dynamic dataset...")
    dynamic_ds = safe_open_dataset(dynamic_file)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Analyze each dataset
    analyze_dataset(target_ds, "Target")
    analyze_dataset(static_ds, "Static")
    analyze_dataset(dynamic_ds, "Dynamic")

    # If any dataset failed to load, we can't proceed with alignment
    if None in [target_ds, static_ds, dynamic_ds]:
        print("Cannot proceed with alignment due to dataset loading errors.")
        return

    # Get station names or identifiers from each dataset
    target_stations = set(target_ds.data_vars)
    static_stations = set(static_ds.index.values)
    dynamic_stations = set(dynamic_ds.station.values)

    print("\nStation counts:")
    print(f"Target stations: {len(target_stations)}")
    print(f"Static stations: {len(static_stations)}")
    print(f"Dynamic stations: {len(dynamic_stations)}")

    # Find common stations
    common_stations = target_stations & dynamic_stations & static_stations
    print(f"\nCommon stations across all datasets: {len(common_stations)}")

    if len(common_stations) == 0:
        print("No common stations found. Cannot proceed with alignment.")
        return

    # Print a few example station names from each dataset
    print("\nExample station names:")
    print("Target:", list(target_stations)[:5])
    print("Static:", list(static_stations)[:5])
    print("Dynamic:", list(dynamic_stations)[:5])

    # Additional analysis can be added here if needed

# Usage
target_file = 'Data/combined_ismn_data/target_data.nc'
static_file = 'Data/combined_ismn_data/static_data.nc'
dynamic_file = 'Data/combined_ismn_data/dynamic_data.nc'
output_dir = 'Data/analysis_output'

analyze_and_visualize_data(target_file, static_file, dynamic_file, output_dir)


# In[ ]:




