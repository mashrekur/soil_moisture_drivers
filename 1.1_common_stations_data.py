import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from tqdm import tqdm

def load_common_stations(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]

def process_static_data(static_data_path, common_stations):
    static_data = xr.open_dataset(static_data_path)
    station_info = pd.read_csv(os.path.join(os.path.dirname(static_data_path), 'complete_stations_info.csv'))
    
    # Filter for common stations
    common_station_info = station_info[station_info['station_name'].isin(common_stations)]
    
    # Define the columns we want to use
    desired_columns = [
        'latitude', 'longitude', 'slope',
        'saturation_1', 'saturation_2',
        'clay_fraction_1', 'clay_fraction_2',
        'organic_carbon_1', 'organic_carbon_2',
        'sand_fraction_1', 'sand_fraction_2',
        'silt_fraction_1', 'silt_fraction_2',
        'bdod', 'cec', 'cfvo', 'clay', 'nitrogen',
        'ocd', 'ocs', 'phh2o', 'sand', 'silt', 'soc',
        'wv0010', 'wv0033', 'wv1500',
        'climate_1_encoded', 'climate_2_encoded',
        'landcover_2020_encoded'
    ]
    
    # Select only the desired columns
    common_static_data = static_data[desired_columns].sel(index=common_station_info.index)
    
    print("Static data columns:", list(common_static_data.data_vars))
    
    return common_static_data, common_station_info

def process_dynamic_data(dynamic_data_dir, common_station_info, start_date, end_date):
    dynamic_files = sorted([f for f in os.listdir(dynamic_data_dir) if f.endswith('.nc')])
    
    all_dynamic_data = []
    for file in tqdm(dynamic_files, desc="Processing dynamic data"):
        ds = xr.open_dataset(os.path.join(dynamic_data_dir, file))
        ds = ds.sel(time=slice(start_date, end_date))
        
        station_data = []
        for _, row in common_station_info.iterrows():
            data = ds.sel(latitude=row['latitude'], longitude=row['longitude'], method='nearest')
            station_data.append(data)
        
        combined_data = xr.concat(station_data, dim='station')
        combined_data['station'] = common_station_info['station_name']
        all_dynamic_data.append(combined_data)
    
    return xr.merge(all_dynamic_data)

def process_target_data(target_data_dir, common_stations, start_date, end_date):
    all_data = {}
    time_index = pd.date_range(start=start_date, end=end_date, freq='H')
    
    for station in tqdm(common_stations, desc="Processing target data"):
        try:
            file_path = os.path.join(target_data_dir, f"{station}_target.nc")
            ds = xr.open_dataset(file_path)
            df = ds.sel(time=slice(start_date, end_date))['soil_moisture'].to_dataframe()
            
            # Reindex to ensure all stations have the same time index
            df = df.reindex(time_index)
            
            all_data[station] = df['soil_moisture']
        except FileNotFoundError:
            print(f"Warning: Target file not found for station {station}")
    
    print(f"Successfully processed {len(all_data)} out of {len(common_stations)} stations")
    
    if not all_data:
        print("Error: No target data could be processed")
        return None
    
    try:
        # Combine all station data
        combined_df = pd.DataFrame(all_data)
        
        # Convert back to xarray Dataset
        combined_ds = combined_df.to_xarray()
        return combined_ds
    except Exception as e:
        print(f"Error combining target data: {str(e)}")
        return None

def main():
    # Input paths
    common_stations_file = 'Data/ismn_analysis_output/common_stations.txt'
    static_data_path = 'Data/static_attributes_model_v3.nc'
    dynamic_data_dir = 'Data/era5_processed_data'
    target_data_dir = 'Data/ismn_target_data'
    
    # Output path
    output_dir = 'Data/combined_ismn_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load common stations
    common_stations = load_common_stations(common_stations_file)
    
    # Set date range
    start_date = '2015-01-01'
    end_date = '2019-12-31'
    
    # Process static data
    print("Processing static data...")
    common_static_data, common_station_info = process_static_data(static_data_path, common_stations)
    common_static_data.to_netcdf(os.path.join(output_dir, 'static_data.nc'))
    print("Static data saved successfully")
    
    # Process dynamic data
    print("Processing dynamic data...")
    dynamic_data = process_dynamic_data(dynamic_data_dir, common_station_info, start_date, end_date)
    dynamic_data.to_netcdf(os.path.join(output_dir, 'dynamic_data.nc'))
    print("Dynamic data saved successfully")
    
    # Process target data
    print("Processing target data...")
    target_data = process_target_data(target_data_dir, common_stations, start_date, end_date)
    if target_data is not None:
        target_data.to_netcdf(os.path.join(output_dir, 'target_data.nc'))
        print("Target data saved successfully")
        print(f"Target data shape: {target_data.soil_moisture.shape}")
        print(f"Time period: {target_data.time.values[0]} to {target_data.time.values[-1]}")
        print(f"Number of stations: {len(target_data.station)}")
        print(f"Number of time steps: {len(target_data.time)}")
        print(f"Percentage of missing data: {(np.isnan(target_data.soil_moisture.values).sum() / target_data.soil_moisture.size * 100):.2f}%")
    else:
        print("Failed to save target data")
    
    print(f"Data processing complete. Files saved in: {output_dir}")

if __name__ == "__main__":
    main()name__ == "__main__":
    main()