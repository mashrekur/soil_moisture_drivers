#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def print_dataset_info(ds, name):
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

def get_target_stations(target_ds):
    return list(target_ds.data_vars)

def find_nearest_point(lat, lon, dynamic_ds):
    distances = np.sqrt((dynamic_ds.latitude - lat)**2 + (dynamic_ds.longitude - lon)**2)
    return distances.argmin().item()

def handle_duplicate_stations(static_ds):
    df = static_ds.to_dataframe().reset_index()
    duplicates = df[df.duplicated('station_name', keep=False)]
    print(f"Found {len(duplicates)} duplicate station names")
    to_keep = duplicates.groupby('station_name').apply(lambda x: x.isnull().sum().idxmin())
    df_unique = df.drop_duplicates('station_name', keep='first')
    return xr.Dataset.from_dataframe(df_unique.set_index('station_name'))

def filter_and_align_datasets(target_ds, static_ds, dynamic_ds):
    target_stations = get_target_stations(target_ds)
    print(f"\nNumber of stations in target dataset: {len(target_stations)}")
    print(f"First few target stations: {target_stations[:5]}")

    static_ds = handle_duplicate_stations(static_ds)
    print_dataset_info(static_ds, "Static (after handling duplicates)")

    static_stations = static_ds.station_name.values
    matching_stations = list(set(target_stations) & set(static_stations))
    print(f"Number of matching stations: {len(matching_stations)}")
    print(f"First few matching stations: {matching_stations[:5]}")

    filtered_static_ds = static_ds.sel(station_name=matching_stations)

    new_dynamic_data = {}
    for var in dynamic_ds.data_vars:
        new_dynamic_data[var] = []

    for station in tqdm(matching_stations, desc="Processing stations"):
        lat = filtered_static_ds.sel(station_name=station).latitude.item()
        lon = filtered_static_ds.sel(station_name=station).longitude.item()
        
        nearest_idx = find_nearest_point(lat, lon, dynamic_ds)
        station_data = dynamic_ds.isel(station=nearest_idx)
        
        for var in dynamic_ds.data_vars:
            new_dynamic_data[var].append(station_data[var].values)

    # Create a new xarray Dataset for dynamic data
    filtered_dynamic_ds = xr.Dataset(
        {var: (('station', 'time'), np.array(new_dynamic_data[var])) for var in dynamic_ds.data_vars},
        coords={
            'station': matching_stations,
            'time': dynamic_ds.time
        }
    )

    filtered_target_ds = target_ds[matching_stations]

    return filtered_target_ds, filtered_static_ds, filtered_dynamic_ds

def analyze_and_visualize_data(target_file, static_file, dynamic_file, output_dir):
    print("Loading target dataset...")
    target_ds = xr.open_dataset(target_file)
    print("Loading static dataset...")
    static_ds = xr.open_dataset(static_file)
    print("Loading dynamic dataset...")
    dynamic_ds = xr.open_dataset(dynamic_file)

    print_dataset_info(target_ds, "Target")
    print_dataset_info(static_ds, "Static")
    print_dataset_info(dynamic_ds, "Dynamic")

    print("\nFiltering and aligning datasets...")
    aligned_target_ds, aligned_static_ds, aligned_dynamic_ds = filter_and_align_datasets(target_ds, static_ds, dynamic_ds)

    print("\n--- Aligned Datasets ---")
    print_dataset_info(aligned_target_ds, "Aligned Target")
    print_dataset_info(aligned_static_ds, "Aligned Static")
    print_dataset_info(aligned_dynamic_ds, "Aligned Dynamic")

    print("\nSaving aligned datasets...")
    os.makedirs(output_dir, exist_ok=True)
    aligned_static_ds.to_netcdf(os.path.join(output_dir, 'aligned_static_data.nc'))
    aligned_dynamic_ds.to_netcdf(os.path.join(output_dir, 'aligned_dynamic_data.nc'))
    print("Aligned datasets saved successfully.")

    target_stations = get_target_stations(aligned_target_ds)
    if len(target_stations) > 0:
        random_station = np.random.choice(target_stations)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))

        aligned_target_ds[random_station].plot(ax=ax1)
        ax1.set_title(f"Soil Moisture Data for Station: {random_station}")
        ax1.set_ylabel("Soil Moisture")

        static_vars = [var for var in aligned_static_ds.data_vars if var != 'station_name']
        aligned_static_ds[static_vars].sel(station_name=random_station).plot(ax=ax2, kind='bar')
        ax2.set_title(f"Static Data for Station: {random_station}")
        ax2.set_xticklabels(static_vars, rotation=90)
        ax2.set_ylabel("Value")

        aligned_dynamic_ds['temperature'].sel(station=random_station).plot(ax=ax3)
        ax3.set_title(f"Temperature Data for Station: {random_station}")
        ax3.set_ylabel("Temperature")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_station_data_{random_station}.png'))
        plt.close()

        print(f"Visualization saved for station: {random_station}")
    else:
        print("No stations available for visualization")

# Usage
target_file = 'Data/processed_ismn_data/processed_target_data.nc'
static_file = 'Data/static_attributes_model_v3.nc'
dynamic_file = 'Data/combined_ismn_data/dynamic_data.nc'
output_dir = 'Data/analysis_output'

analyze_and_visualize_data(target_file, static_file, dynamic_file, output_dir)


# In[5]:




