#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import xarray as xr
from tqdm import tqdm
import logging
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_datetime(date, time):
    try:
        return pd.to_datetime(date + " " + time, format="%Y/%m/%d %H:%M")
    except ValueError:
        return pd.NaT

def load_ismn_data(ismn_dir):
    station_data = {}
    for root, dirs, files in os.walk(ismn_dir):
        for file in tqdm(files, desc="Processing ISMN files"):
            if file.endswith(".stm") and not file.startswith("._"):
                file_path = os.path.join(root, file)
                try:
                    # Try different encodings
                    encodings = ['utf-8', 'iso-8859-1', 'cp1252']
                    for encoding in encodings:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                df = pd.read_csv(file_path, sep=r'\s+', header=None, skiprows=1,
                                                 names=["date", "time", "soil_moisture", "qc_flag", "provider_flag"],
                                                 encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        raise UnicodeDecodeError(f"Failed to decode {file} with any of the attempted encodings")

                    df["datetime"] = df.apply(lambda row: parse_datetime(row["date"], row["time"]), axis=1)
                    df = df.dropna(subset=["datetime"])
                    
                    station_name = os.path.basename(os.path.dirname(file_path))
                    if station_name not in station_data:
                        station_data[station_name] = df[["datetime", "soil_moisture"]]
                    else:
                        station_data[station_name] = pd.concat([station_data[station_name], df[["datetime", "soil_moisture"]]])
                except Exception as e:
                    logging.error(f"Error processing {file}: {str(e)}")
    
    # Sort data by datetime for each station
    for station in station_data:
        station_data[station] = station_data[station].sort_values("datetime").reset_index(drop=True)
    
    return station_data

def create_target_dataset(ismn_data, stations_info):
    target_datasets = {}
    for station, data in tqdm(ismn_data.items(), desc="Creating target datasets"):
        if station in stations_info['station_name'].values:
            station_info = stations_info[stations_info['station_name'] == station].iloc[0]
            da = xr.DataArray(
                data=data['soil_moisture'].values,
                dims=['time'],
                coords={
                    'time': data['datetime'],
                    'latitude': station_info['latitude'],
                    'longitude': station_info['longitude']
                },
                name='soil_moisture'
            )
            target_datasets[station] = da
        else:
            logging.warning(f"Station {station} not found in stations_info")
    return target_datasets

def main():
    ismn_dir = "data/ismn_2000_2023/"
    stations_info_file = 'Data/complete_stations_info.csv'
    output_dir = 'Data/ismn_target_data'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load station information
    stations_info = pd.read_csv(stations_info_file)
    
    # Load ISMN data
    logging.info("Loading ISMN data...")
    ismn_data = load_ismn_data(ismn_dir)
    
    # Create target datasets
    logging.info("Creating target datasets...")
    target_datasets = create_target_dataset(ismn_data, stations_info)
    
    # Save target datasets
    logging.info("Saving target datasets...")
    for station, dataset in tqdm(target_datasets.items(), desc="Saving datasets"):
        output_file = os.path.join(output_dir, f"{station}_target.nc")
        dataset.to_netcdf(output_file)
    
    logging.info(f"Processed {len(target_datasets)} stations")
    logging.info(f"Target datasets saved in {output_dir}")

if __name__ == "__main__":
    main()


# In[ ]:




