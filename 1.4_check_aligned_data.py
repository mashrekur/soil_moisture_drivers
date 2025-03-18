#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def load_data(static_file, dynamic_file, target_file):
    static_ds = xr.open_dataset(static_file)
    dynamic_ds = xr.open_dataset(dynamic_file)
    target_ds = xr.open_dataset(target_file)
    return static_ds, dynamic_ds, target_ds

def find_nearest_point(lat, lon, dynamic_ds):
    distances = np.sqrt((dynamic_ds.latitude - lat)**2 + (dynamic_ds.longitude - lon)**2)
    return distances.argmin().item()

def handle_duplicate_stations(static_ds):
    # Convert to DataFrame for easier handling
    df = static_ds.to_dataframe().reset_index()
    
    # Identify duplicates
    duplicates = df[df.duplicated('station_name', keep=False)]
    print(f"Found {len(duplicates)} duplicate station names")
    
    # For each set of duplicates, keep the one with the most non-null values
    to_keep = duplicates.groupby('station_name').apply(lambda x: x.isnull().sum().idxmin())
    
    # Remove duplicates from the original DataFrame
    df_unique = df.drop_duplicates('station_name', keep='first')
    
    # Convert back to xarray Dataset
    return xr.Dataset.from_dataframe(df_unique.set_index('station_name'))

def filter_and_align_datasets(target_ds, static_ds, dynamic_ds):
    target_stations = list(target_ds.data_vars)
    print(f"\nNumber of stations in target dataset: {len(target_stations)}")

    # Handle duplicate stations in static dataset
    static_ds = handle_duplicate_stations(static_ds)
    
    static_stations = static_ds.station_name.values
    matching_stations = list(set(target_stations) & set(static_stations))
    print(f"Number of matching stations: {len(matching_stations)}")

    filtered_static_ds = static_ds.sel(station_name=matching_stations)

    new_dynamic_data = {}
    lats = []
    lons = []
    for station in tqdm(matching_stations, desc="Processing stations"):
        lat = filtered_static_ds.sel(station_name=station).latitude.item()
        lon = filtered_static_ds.sel(station_name=station).longitude.item()
        
        nearest_idx = find_nearest_point(lat, lon, dynamic_ds)
        station_data = dynamic_ds.isel(station=nearest_idx)
        
        for var in dynamic_ds.data_vars:
            if var not in new_dynamic_data:
                new_dynamic_data[var] = []
            new_dynamic_data[var].append(station_data[var].values)
        
        lats.append(lat)
        lons.append(lon)

    # Create a new xarray Dataset for dynamic data
    filtered_dynamic_ds = xr.Dataset(
        {var: (('station', 'time'), np.array(new_dynamic_data[var])) for var in dynamic_ds.data_vars},
        coords={
            'station': matching_stations,
            'time': dynamic_ds.time,
            'latitude': ('station', np.array(lats)),
            'longitude': ('station', np.array(lons))
        }
    )

    filtered_target_ds = target_ds[matching_stations]

    return filtered_target_ds, filtered_static_ds, filtered_dynamic_ds

def main():
    static_file = 'Data/static_attributes_model_v3.nc'
    dynamic_file = 'Data/combined_ismn_data/dynamic_data.nc'
    target_file = 'Data/processed_ismn_data/processed_target_data.nc'
    output_dir = 'Data/analysis_output'

    static_ds, dynamic_ds, target_ds = load_data(static_file, dynamic_file, target_file)

    aligned_target_ds, aligned_static_ds, aligned_dynamic_ds = filter_and_align_datasets(target_ds, static_ds, dynamic_ds)

    os.makedirs(output_dir, exist_ok=True)
    aligned_static_ds.to_netcdf(os.path.join(output_dir, 'aligned_static_data.nc'))
    aligned_dynamic_ds.to_netcdf(os.path.join(output_dir, 'aligned_dynamic_data.nc'))
    aligned_target_ds.to_netcdf(os.path.join(output_dir, 'aligned_target_data.nc'))

    print("Aligned datasets saved successfully.")

if __name__ == "__main__":
    main()


# In[5]:


import xarray as xr
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def analyze_and_handle_duplicate_stations(static_ds):
    df = static_ds.to_dataframe().reset_index()
    duplicates = df[df.duplicated('station_name', keep=False)].sort_values('station_name')
    print(f"Found {len(duplicates)} duplicate station entries")
    print(f"Number of unique stations with duplicates: {duplicates['station_name'].nunique()}")
    
    resolved_df = df.drop_duplicates('station_name', keep=False)  # Keep non-duplicates as is
    
    for station in duplicates['station_name'].unique():
        station_entries = duplicates[duplicates['station_name'] == station]
        print(f"\nAnalyzing duplicate entries for station: {station}")
        print(f"Number of entries: {len(station_entries)}")
        
        if station_entries['latitude'].nunique() > 1 or station_entries['longitude'].nunique() > 1:
            print("Different coordinates detected. Renaming stations.")
            for idx, row in station_entries.iterrows():
                new_name = f"{station}_{row['latitude']}_{row['longitude']}"
                row['station_name'] = new_name
                resolved_df = resolved_df.append(row, ignore_index=True)
        else:
            print("Coordinates are the same. Keeping one entry.")
            resolved_df = resolved_df.append(station_entries.iloc[0], ignore_index=True)
    
    # Handle case sensitivity in station names
    resolved_df['station_name'] = resolved_df['station_name'].str.lower()
    resolved_df = resolved_df.drop_duplicates('station_name', keep='first')
    
    return xr.Dataset.from_dataframe(resolved_df.set_index('station_name'))


def load_data(static_file, dynamic_file, target_file):
    static_ds = xr.open_dataset(static_file)
    dynamic_ds = xr.open_dataset(dynamic_file)
    target_ds = xr.open_dataset(target_file)
    return static_ds, dynamic_ds, target_ds

def find_nearest_point(lat, lon, dynamic_ds):
    distances = np.sqrt((dynamic_ds.latitude - lat)**2 + (dynamic_ds.longitude - lon)**2)
    return distances.argmin().item()

def analyze_duplicate_stations(static_ds):
    # Convert to DataFrame for easier handling
    df = static_ds.to_dataframe().reset_index()
    
    # Identify duplicates
    duplicates = df[df.duplicated('station_name', keep=False)]
    print(f"Found {len(duplicates)} duplicate station entries")
    print(f"Number of unique stations with duplicates: {duplicates['station_name'].nunique()}")
    
    # Analyze differences in duplicate entries
    for station in duplicates['station_name'].unique():
        station_entries = duplicates[duplicates['station_name'] == station]
        print(f"\nAnalyzing duplicate entries for station: {station}")
        print(f"Number of entries: {len(station_entries)}")
        
        # Check if lat/lon are the same
        if station_entries['latitude'].nunique() > 1 or station_entries['longitude'].nunique() > 1:
            print("Warning: Latitude or Longitude differs between entries")
            print(station_entries[['latitude', 'longitude']])
        
        # Check for differences in other columns
        diff_columns = station_entries.columns[station_entries.nunique() > 1]
        if len(diff_columns) > 0:
            print("Columns with different values:")
            for col in diff_columns:
                if col not in ['latitude', 'longitude', 'station_name']:
                    print(f"  {col}: {station_entries[col].unique()}")
    
    # For each set of duplicates, keep the one with the most non-null values
    to_keep = duplicates.groupby('station_name').apply(lambda x: x.isnull().sum().idxmin())
    
    # Remove duplicates from the original DataFrame
    df_unique = df.drop_duplicates('station_name', keep='first')
    
    # Convert back to xarray Dataset
    return xr.Dataset.from_dataframe(df_unique.set_index('station_name'))

def filter_and_align_datasets(target_ds, static_ds, dynamic_ds):
    target_stations = list(target_ds.data_vars)
    print(f"\nNumber of stations in target dataset: {len(target_stations)}")

    # Analyze and handle duplicate stations in static dataset
    static_ds = analyze_duplicate_stations(static_ds)
    
    static_stations = static_ds.station_name.values
    matching_stations = list(set(target_stations) & set(static_stations))
    print(f"Number of matching stations: {len(matching_stations)}")

    filtered_static_ds = static_ds.sel(station_name=matching_stations)

    new_dynamic_data = {}
    lats = []
    lons = []
    for station in tqdm(matching_stations, desc="Processing stations"):
        lat = filtered_static_ds.sel(station_name=station).latitude.item()
        lon = filtered_static_ds.sel(station_name=station).longitude.item()
        
        nearest_idx = find_nearest_point(lat, lon, dynamic_ds)
        station_data = dynamic_ds.isel(station=nearest_idx)
        
        for var in dynamic_ds.data_vars:
            if var not in new_dynamic_data:
                new_dynamic_data[var] = []
            new_dynamic_data[var].append(station_data[var].values)
        
        lats.append(lat)
        lons.append(lon)

    # Create a new xarray Dataset for dynamic data
    filtered_dynamic_ds = xr.Dataset(
        {var: (('station', 'time'), np.array(new_dynamic_data[var])) for var in dynamic_ds.data_vars},
        coords={
            'station': matching_stations,
            'time': dynamic_ds.time,
            'latitude': ('station', np.array(lats)),
            'longitude': ('station', np.array(lons))
        }
    )

    filtered_target_ds = target_ds[matching_stations]

    return filtered_target_ds, filtered_static_ds, filtered_dynamic_ds

def main():
    static_file = 'Data/static_attributes_model_v3.nc'
    dynamic_file = 'Data/combined_ismn_data/dynamic_data.nc'
    target_file = 'Data/processed_ismn_data/processed_target_data.nc'
    output_dir = 'Data/analysis_output'

    static_ds, dynamic_ds, target_ds = load_data(static_file, dynamic_file, target_file)

    aligned_target_ds, aligned_static_ds, aligned_dynamic_ds = filter_and_align_datasets(target_ds, static_ds, dynamic_ds)

    os.makedirs(output_dir, exist_ok=True)
    aligned_static_ds.to_netcdf(os.path.join(output_dir, 'aligned_static_data.nc'))
    aligned_dynamic_ds.to_netcdf(os.path.join(output_dir, 'aligned_dynamic_data.nc'))
    aligned_target_ds.to_netcdf(os.path.join(output_dir, 'aligned_target_data.nc'))

    print("Aligned datasets saved successfully.")

if __name__ == "__main__":
    main()


# In[13]:


import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import os

def plot_dynamic_data(ds, variable, time_points, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5)})
    axes = axes.flatten()

    # Check how coordinates are stored in the dataset
    if 'longitude' in ds.coords:
        lons = ds.longitude
        lats = ds.latitude
    elif 'lon' in ds.coords:
        lons = ds.lon
        lats = ds.lat
    else:
        raise ValueError("Couldn't find longitude and latitude coordinates in the dataset")

    for i, time_point in enumerate(time_points):
        ax = axes[i]
        data = ds[variable].sel(time=time_point)
        
        ax.add_feature(cfeature.STATES)
        ax.set_extent([-125, -66.5, 24, 49], crs=ccrs.PlateCarree())

        scatter = ax.scatter(lons, lats, c=data, 
                             cmap='viridis', transform=ccrs.PlateCarree(),
                             s=20, alpha=0.7)
        
        # Convert numpy.datetime64 to pandas.Timestamp for strftime
        time_str = pd.Timestamp(time_point).strftime("%Y-%m-%d")
        ax.set_title(f'{variable} on {time_str}')
        plt.colorbar(scatter, ax=ax, label=variable, shrink=0.6)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{variable}_dynamic_map.png', dpi=300, bbox_inches='tight')
    plt.close()

# Load the data
dynamic_file = 'Data/analysis_output/aligned_dynamic_data.nc'
dynamic_ds = xr.open_dataset(dynamic_file)

# Select 4 time points (e.g., one for each season in the middle year)
time_points = [
    np.datetime64('2017-01-15'),
    np.datetime64('2017-04-15'),
    np.datetime64('2017-07-15'),
    np.datetime64('2017-10-15')
]

output_dir = 'Data/visualization_output'
os.makedirs(output_dir, exist_ok=True)

# Plot for each dynamic variable
for variable in ['temperature', 'precipitation', 'solar_radiation']:
    plot_dynamic_data(dynamic_ds, variable, time_points, output_dir)

print("Dynamic data visualizations saved in:", output_dir)


# In[9]:


import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

def plot_static_data(ds, variables, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5)})
    axes = axes.flatten()

    for i, variable in enumerate(variables):
        ax = axes[i]
        data = ds[variable]
        
        ax.add_feature(cfeature.STATES)
        ax.set_extent([-125, -66.5, 24, 49], crs=ccrs.PlateCarree())

        scatter = ax.scatter(ds.longitude, ds.latitude, c=data, 
                             cmap='viridis', transform=ccrs.PlateCarree(),
                             s=20, alpha=0.7)
        
        ax.set_title(variable)
        plt.colorbar(scatter, ax=ax, label=variable, shrink=0.6)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/static_variables_map.png', dpi=300, bbox_inches='tight')
    plt.close()

# Load the data
static_file = 'Data/analysis_output/aligned_static_data.nc'
static_ds = xr.open_dataset(static_file)

output_dir = 'Data/visualization_output'
os.makedirs(output_dir, exist_ok=True)

# Select 4 static variables
static_variables = ['slope', 'clay_fraction_1', 'sand_fraction_1', 'organic_carbon_1']

# Plot static variables
plot_static_data(static_ds, static_variables, output_dir)

print("Static data visualization saved in:", output_dir)


# In[1]:


import xarray as xr
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def add_coordinates_if_missing(ds, static_file):
    if 'latitude' not in ds.coords or 'longitude' not in ds.coords:
        print("Adding latitude and longitude coordinates from static data...")
        static_ds = xr.open_dataset(static_file)
        ds = ds.assign_coords(latitude=('station', static_ds.latitude.values))
        ds = ds.assign_coords(longitude=('station', static_ds.longitude.values))
    else:
        print("Latitude and longitude coordinates already present in the dataset.")
    return ds

def plot_target_data(ds, time_points, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5)})
    axes = axes.flatten()

    for i, time_point in enumerate(time_points):
        ax = axes[i]
        data = ds.sel(index=time_point).to_array().values.squeeze()
        
        ax.add_feature(cfeature.STATES)
        ax.set_extent([-125, -66.5, 24, 49], crs=ccrs.PlateCarree())

        scatter = ax.scatter(ds.longitude, ds.latitude, c=data, 
                             cmap='YlGnBu', transform=ccrs.PlateCarree(),
                             s=20, alpha=0.7)
        
        time_str = pd.Timestamp(time_point).strftime("%Y-%m-%d")
        ax.set_title(f'Soil Moisture on {time_str}')
        plt.colorbar(scatter, ax=ax, label='Soil Moisture', shrink=0.6)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/target_data_map.png', dpi=300, bbox_inches='tight')
    plt.close()

# Load the data
target_file = 'Data/analysis_output/aligned_target_data.nc'
static_file = 'Data/analysis_output/aligned_static_data.nc'
target_ds = xr.open_dataset(target_file)

# Add coordinates if they're missing
target_ds = add_coordinates_if_missing(target_ds, static_file)

# Save the updated target dataset
output_dir = 'Data/analysis_output'
os.makedirs(output_dir, exist_ok=True)
new_target_file = os.path.join(output_dir, 'aligned_target_data_with_coords.nc')
target_ds.to_netcdf(new_target_file)
print(f"Updated aligned target dataset saved to: {new_target_file}")

# Create visualization
vis_output_dir = 'Data/visualization_output'
os.makedirs(vis_output_dir, exist_ok=True)

# Select 4 time points (e.g., one for each season in the middle year)
time_points = [
    np.datetime64('2017-01-15'),
    np.datetime64('2017-04-15'),
    np.datetime64('2017-07-15'),
    np.datetime64('2017-10-15')
]

# Plot target data
plot_target_data(target_ds, time_points, vis_output_dir)

print("Target data visualization saved in:", vis_output_dir)

# Print some information about the updated dataset
print("\nUpdated Target Dataset Information:")
print(target_ds.info())
print("\nCoordinates:")
print(target_ds.coords)
print("\nData Variables:")
print(target_ds.data_vars)


# In[5]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

def reshape_and_save_target_data(input_file, output_file):
    # Load the data
    target_ds = xr.open_dataset(input_file)

    # Reshape the dataset
    reshaped_ds = target_ds.to_array(dim='station', name='soil_moisture').to_dataset()

    # Rename 'index' to 'time' for clarity
    reshaped_ds = reshaped_ds.rename({'index': 'time'})

    # Save the reshaped dataset
    reshaped_ds.to_netcdf(output_file)

    return reshaped_ds

def plot_reshaped_data(ds, output_dir):
    # Select a few time points for visualization
    time_points = [
        np.datetime64('2017-01-15'),
        np.datetime64('2017-04-15'),
        np.datetime64('2017-07-15'),
        np.datetime64('2017-10-15')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16), subplot_kw={'projection': ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5)})
    axes = axes.flatten()

    for i, time_point in enumerate(time_points):
        ax = axes[i]
        data = ds.soil_moisture.sel(time=time_point)
        
        ax.add_feature(cfeature.STATES)
        ax.set_extent([-125, -66.5, 24, 49], crs=ccrs.PlateCarree())

        scatter = ax.scatter(ds.longitude, ds.latitude, c=data, 
                             cmap='YlGnBu', transform=ccrs.PlateCarree(),
                             s=20, alpha=0.7)
        
        ax.set_title(f'Soil Moisture on {time_point}')
        plt.colorbar(scatter, ax=ax, label='Soil Moisture', shrink=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reshaped_target_data_map.png'), dpi=300, bbox_inches='tight')
    plt.close()

# File paths
input_file = 'Data/analysis_output/aligned_target_data_with_coords.nc'
output_file = 'Data/analysis_output/reshaped_aligned_target_data.nc'
output_dir = 'Data/visualization_output'

# Reshape and save the data
reshaped_ds = reshape_and_save_target_data(input_file, output_file)

print("Reshaped dataset information:")
print(reshaped_ds)

print("\nCoordinates:")
print(reshaped_ds.coords)

print("\nData Variables:")
print(reshaped_ds.data_vars)

# Create a visualization of the reshaped data
os.makedirs(output_dir, exist_ok=True)
plot_reshaped_data(reshaped_ds, output_dir)

print(f"\nReshaped target data saved to: {output_file}")
print(f"Visualization saved to: {output_dir}/reshaped_target_data_map.png")


# In[7]:


# diagnose_data.py
import xarray as xr
import logging
import os

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def examine_dataset(file_path, dataset_name):
    """Examine the structure of a dataset in detail."""
    logging.info(f"\nExamining {dataset_name} from: {file_path}")
    try:
        ds = xr.open_dataset(file_path)
        
        logging.info(f"\n{dataset_name} Dataset Structure:")
        logging.info("-" * 50)
        logging.info(f"Dimensions: {dict(ds.dims)}")
        logging.info("\nCoordinates:")
        for coord in ds.coords:
            logging.info(f"  {coord}: {ds.coords[coord].shape}")
        
        logging.info("\nData Variables:")
        for var in ds.data_vars:
            logging.info(f"  {var}:")
            logging.info(f"    Shape: {ds[var].shape}")
            logging.info(f"    Dims: {ds[var].dims}")
            logging.info(f"    Dtype: {ds[var].dtype}")
        
        # Try to access first few values
        logging.info("\nFirst few values of each variable:")
        for var in ds.data_vars:
            try:
                logging.info(f"  {var}: {ds[var].values.flatten()[:5]} ...")
            except Exception as e:
                logging.info(f"  {var}: Could not access values - {str(e)}")
        
        ds.close()
        return ds
        
    except Exception as e:
        logging.error(f"Error examining {dataset_name}: {str(e)}")
        return None

def main():
    setup_logging()
    
    # File paths
    data_dir = 'Data/analysis_output'
    static_file = os.path.join(data_dir, 'aligned_static_data.nc')
    dynamic_file = os.path.join(data_dir, 'aligned_dynamic_data.nc')
    target_file = os.path.join(data_dir, 'reshaped_aligned_target_data.nc')
    
    # Examine each dataset
    static_ds = examine_dataset(static_file, "Static")
    dynamic_ds = examine_dataset(dynamic_file, "Dynamic")
    target_ds = examine_dataset(target_file, "Target")
    
    # Check for alignment
    if all([static_ds, dynamic_ds, target_ds]):
        logging.info("\nChecking dataset alignment:")
        
        # Get all dimension names
        all_dims = set()
        for ds in [static_ds, dynamic_ds, target_ds]:
            all_dims.update(ds.dims)
        
        logging.info(f"\nAll dimensions found across datasets: {all_dims}")
        
        # Compare station identifiers if they exist
        for ds_name, ds in [("Static", static_ds), ("Dynamic", dynamic_ds), ("Target", target_ds)]:
            logging.info(f"\n{ds_name} dataset index/station information:")
            if 'station' in ds.dims:
                logging.info(f"  Has 'station' dimension with length {len(ds.station)}")
            if 'index' in ds.dims:
                logging.info(f"  Has 'index' dimension with length {len(ds.index)}")
            if hasattr(ds, 'station_name'):
                logging.info(f"  Has station_name variable/coordinate")

if __name__ == "__main__":
    main()


# In[9]:


import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_static_data(file_path):
    """Analyze static data for NaN values and patterns."""
    print(f"Analyzing static data from: {file_path}")
    
    # Load the data
    ds = xr.open_dataset(file_path)
    
    # Convert to DataFrame for easier analysis
    df = ds.to_dataframe()
    
    print("\n1. Basic Information:")
    print("-" * 50)
    print(f"Number of stations: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    
    print("\n2. NaN Analysis:")
    print("-" * 50)
    nan_counts = df.isna().sum()
    print("NaN counts per column:")
    for col, count in nan_counts.items():
        if count > 0:
            print(f"{col}: {count} NaNs ({count/len(df)*100:.2f}%)")
    
    print("\n3. Categorical Variables Analysis:")
    print("-" * 50)
    categorical_cols = [col for col in df.columns if 'encoded' in col]
    raw_cat_cols = ['climate_1', 'climate_2', 'landcover_2020']
    
    print("Encoded categorical variables:")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"Unique values: {sorted(df[col].unique())}")
        print(f"NaN count: {df[col].isna().sum()}")
        
        # Check corresponding raw values
        raw_col = col.replace('_encoded', '')
        if raw_col in df.columns:
            print(f"Raw values for NaN encodings:")
            print(df[df[col].isna()][raw_col].value_counts())
    
    print("\n4. Correlation with NaN Presence:")
    print("-" * 50)
    # Create binary NaN indicators for each column
    nan_indicators = pd.DataFrame()
    for col in df.columns:
        if df[col].isna().any():
            nan_indicators[f'{col}_has_nan'] = df[col].isna().astype(int)
    
    if not nan_indicators.empty:
        corr = nan_indicators.corr()
        print("Correlations between NaN occurrences:")
        print(corr[corr > 0.5].stack().drop_duplicates())
    
    print("\n5. Row-wise NaN Analysis:")
    print("-" * 50)
    nan_counts_per_row = df.isna().sum(axis=1)
    print("Distribution of NaNs across rows:")
    print(nan_counts_per_row.value_counts().sort_index())
    
    # Plot NaN patterns
    plt.figure(figsize=(15, 8))
    sns.heatmap(df.isna(), yticklabels=False, cbar=False)
    plt.title('NaN Pattern Visualization')
    plt.xlabel('Features')
    plt.ylabel('Stations')
    plt.savefig('nan_pattern.png')
    plt.close()
    
    # Save problematic rows to CSV
    problematic_rows = df[df.isna().any(axis=1)]
    if not problematic_rows.empty:
        problematic_rows.to_csv('problematic_stations.csv')
        print(f"\nSaved {len(problematic_rows)} problematic stations to 'problematic_stations.csv'")
    
    return df, nan_indicators

if __name__ == "__main__":
    static_data_path = 'Data/analysis_output/aligned_static_data.nc'
    df, nan_indicators = analyze_static_data(static_data_path)
    
    # Additional analysis for encoded variables
    encoded_vars = [col for col in df.columns if '_encoded' in col]
    raw_vars = [col.replace('_encoded', '') for col in encoded_vars]
    
    print("\n6. Encoded vs Raw Values Analysis:")
    print("-" * 50)
    for encoded, raw in zip(encoded_vars, raw_vars):
        if raw in df.columns:
            print(f"\nAnalyzing {encoded} vs {raw}:")
            mapping = df.groupby(raw)[encoded].first().dropna()
            print("Value mapping:")
            print(mapping)
            
            # Check for inconsistencies
            for raw_val in df[raw].unique():
                encodings = df[df[raw] == raw_val][encoded].unique()
                if len(encodings) > 1:
                    print(f"Warning: Raw value '{raw_val}' has multiple encodings: {encodings}")
    
    print("\n7. Feature Statistics:")
    print("-" * 50)
    print(df.describe())
    
    # Save complete analysis to file
    with open('static_data_analysis.txt', 'w') as f:
        f.write("Static Data Analysis Report\n")
        f.write("==========================\n")
        f.write(f"\nShape: {df.shape}\n")
        f.write("\nMissing Values Summary:\n")
        f.write(df.isna().sum().to_string())
        f.write("\n\nFeature Statistics:\n")
        f.write(df.describe().to_string())


# In[ ]:




