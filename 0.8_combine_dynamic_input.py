#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import xarray as xr
import numpy as np
from tqdm import tqdm
import logging
import cdsapi
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_era5_data(c, stations_info, year, output_dir):
    raw_file_name = os.path.join(output_dir, f'era5_raw_data_{year}.nc')
    
    if os.path.exists(raw_file_name):
        logging.info(f"Raw data for year {year} already exists. Skipping download.")
        return raw_file_name

    try:
        # Calculate the bounding box for all stations
        min_lat = stations_info['latitude'].min() - 0.1
        max_lat = stations_info['latitude'].max() + 0.1
        min_lon = stations_info['longitude'].min() - 0.1
        max_lon = stations_info['longitude'].max() + 0.1
        
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    '2m_temperature', 'total_precipitation',
                    'surface_solar_radiation_downwards'
                ],
                'year': f'{year}',
                'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
                'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
                'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
                'area': [
                    max_lat, min_lon, min_lat, max_lon,
                ],
                'format': 'netcdf',
            },
            raw_file_name)
        
        logging.info(f"Downloaded and saved raw ERA5 data for year {year} to {raw_file_name}")
        return raw_file_name
    except Exception as e:
        logging.error(f"Error downloading ERA5 data for year {year}: {str(e)}")
        return None

def process_era5_data(raw_file_name, output_dir, year):
    try:
        ds = xr.open_dataset(raw_file_name)
        
        # Convert precipitation from m to mm
        if 'tp' in ds:
            ds['tp'] = ds['tp'] * 1000
            ds = ds.rename({'tp': 'precipitation'})
        
        # Convert temperature from K to °C
        if 't2m' in ds:
            ds['t2m'] = ds['t2m'] - 273.15
            ds = ds.rename({'t2m': 'temperature'})
        
        # Process solar radiation
        if 'ssrd' in ds:
            # Convert from J/m^2 (accumulated over hour) to W/m^2 (average over hour)
            ds['ssrd'] = ds['ssrd'] / 3600
            ds = ds.rename({'ssrd': 'solar_radiation'})
        
        processed_file_name = os.path.join(output_dir, f'era5_processed_data_{year}.nc')
        ds.to_netcdf(processed_file_name)
        ds.close()
        
        logging.info(f"Processed and saved ERA5 data for year {year} to {processed_file_name}")
        return processed_file_name
    except Exception as e:
        logging.error(f"Error processing ERA5 data for year {year}: {str(e)}")
        return None

def main():
    stations_info_file = 'Data/complete_stations_info.csv'
    raw_output_dir = 'Data/era5_raw_data'
    processed_output_dir = 'Data/era5_processed_data'
    start_year = 2000
    end_year = 2023
    
    os.makedirs(raw_output_dir, exist_ok=True)
    os.makedirs(processed_output_dir, exist_ok=True)
    
    # Load station information
    stations_info = pd.read_csv(stations_info_file)
    
    c = cdsapi.Client()
    
    # Download and process data year by year
    for year in tqdm(range(start_year, end_year + 1), desc="Processing years"):
        logging.info(f"Processing data for year {year}")
        
        # Download raw data
        raw_file_name = download_era5_data(c, stations_info, year, raw_output_dir)
        
        if raw_file_name:
            # Process the raw data
            processed_file_name = process_era5_data(raw_file_name, processed_output_dir, year)
            
            if processed_file_name:
                logging.info(f"Successfully processed data for year {year}")
            else:
                logging.warning(f"Failed to process data for year {year}")
        else:
            logging.warning(f"Failed to download data for year {year}")
        
        time.sleep(5)  # Add a delay between years to avoid overwhelming the API
    
    logging.info(f"Raw ERA5 data saved in {raw_output_dir}")
    logging.info(f"Processed ERA5 data saved in {processed_output_dir}")

if __name__ == "__main__":
    main()


# In[ ]:




