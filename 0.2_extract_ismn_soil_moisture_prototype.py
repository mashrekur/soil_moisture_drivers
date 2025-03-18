#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import csv
from scipy import stats
import numpy as np


# In[2]:


main_dir = "data/ismn_2000_2023/"
usda_ars_dir = "data/ismn_2000_2023/USDA-ARS"


# In[3]:


# Initialize a dictionary to store the time series data for each station
station_data = {}

for root, dirs, files in os.walk(usda_ars_dir):
    for file in files:
        if file.endswith(".stm") and "average" in file:
            file_path = os.path.join(root, file)
            
            try:
                parts = file.split("_")
                data_source = parts[0]
                station_name = parts[2]
                
                df = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=1,
                                 names=["date", "time", "soil_moisture", "qc_flag", "provider_flag"])
                
                df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y/%m/%d %H:%M")
                
                station_data[(data_source, station_name)] = df[["datetime", "soil_moisture"]]
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

# Print summary of the collected data
print(f"Total stations processed: {len(station_data)}")
for station, data in station_data.items():
    print(f"\nStation: {station}")
    print(f"Data points: {len(data)}")
    print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
    print(f"Sample data:")
    print(data.head())
    print("-" * 50)


# In[4]:


# Plot time series for all stations
plt.figure(figsize=(15, 10))
for station, data in station_data.items():
    plt.plot(data['datetime'], data['soil_moisture'], label=station[1])

plt.title("Soil Moisture Time Series for USDA-ARS Stations")
plt.xlabel("Date")
plt.ylabel("Soil Moisture")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[9]:


def load_ars_data(directory):
    station_data = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".stm") and "average" in file:
                file_path = os.path.join(root, file)
                parts = file.split("_")
                station_name = parts[2]
                df = pd.read_csv(file_path, delim_whitespace=True, header=None, skiprows=1,
                                 names=["date", "time", "soil_moisture", "qc_flag", "provider_flag"])
                df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y/%m/%d %H:%M")
                station_data[station_name] = df[["datetime", "soil_moisture"]]
    return station_data

def plot_histograms(station_data):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    for i, (station, data) in enumerate(station_data.items()):
        axes[i].hist(data['soil_moisture'], bins=50, edgecolor='black')
        axes[i].set_title(f"{station} Soil Moisture Distribution")
        axes[i].set_xlabel("Soil Moisture")
        axes[i].set_ylabel("Frequency")
        mu, std = stats.norm.fit(data['soil_moisture'])
        xmin, xmax = axes[i].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        axes[i].plot(x, p * len(data) * (xmax-xmin) / 50, 'k', linewidth=2)
    plt.tight_layout()
    plt.show()

def extract_recession_curves(station_data, min_duration=48, min_decrease_percent=5):
    recession_curves = {station: [] for station in station_data.keys()}
    for station, data in station_data.items():
        i = 0
        while i < len(data) - min_duration:
            window = data['soil_moisture'].iloc[i:i+min_duration]
            initial_moisture = window.iloc[0]
            final_moisture = window.iloc[-1]
            decrease_percent = (initial_moisture - final_moisture) / initial_moisture * 100
            
            if decrease_percent >= min_decrease_percent and all(window.diff().dropna() <= 0):
                end = i + min_duration
                while end < len(data) and data['soil_moisture'].iloc[end] <= data['soil_moisture'].iloc[end-1]:
                    end += 1
                
                recession_curves[station].append({
                    'start_date': data['datetime'].iloc[i],
                    'end_date': data['datetime'].iloc[end-1],
                    'duration': end - i,
                    'initial_moisture': initial_moisture,
                    'final_moisture': data['soil_moisture'].iloc[end-1],
                    'decrease_percent': (initial_moisture - data['soil_moisture'].iloc[end-1]) / initial_moisture * 100,
                    'curve': data['soil_moisture'].iloc[i:end].values
                })
                i = end
            else:
                i += 1
    
    return recession_curves

def calculate_stats(recession_curves):
    stats = {}
    for station, curves in recession_curves.items():
        if curves:
            stats[station] = {
                'num_curves': len(curves),
                'mean_initial': np.mean([curve['initial_moisture'] for curve in curves]),
                'mean_final': np.mean([curve['final_moisture'] for curve in curves]),
                'mean_duration': np.mean([curve['duration'] for curve in curves]),
                'mean_decrease_percent': np.mean([curve['decrease_percent'] for curve in curves]),
                'std_initial': np.std([curve['initial_moisture'] for curve in curves]),
                'std_final': np.std([curve['final_moisture'] for curve in curves]),
                'std_duration': np.std([curve['duration'] for curve in curves]),
                'std_decrease_percent': np.std([curve['decrease_percent'] for curve in curves])
            }
    return stats


# In[10]:


# Main execution
usda_ars_dir = "data/ismn_2000_2023/USDA-ARS"
station_data = load_ars_data(usda_ars_dir)


# In[11]:


# Plot histograms
plot_histograms(station_data)


# In[12]:


# Extract recession curves
recession_curves = extract_recession_curves(station_data)


# In[13]:


recession_curves


# In[15]:


# Calculate statistics
stats = calculate_stats(recession_curves)

# Print statistics and store in CSV
stats_df = pd.DataFrame(stats).T
print(stats_df)
stats_df.to_csv('recession_curve_stats.csv')


# In[20]:


# Plot all recession curves
plt.figure(figsize=(15, 10))
for station, curves in recession_curves.items():
    for curve_data in curves:
        curve = curve_data['curve']
        if len(curve) > 200:
            curve = curve[:200]  # Truncate curves longer than 200 hours
        plt.plot(range(len(curve)), curve, alpha=0.1, label=station)

plt.title("Soil Moisture Recession Curves (48+ hours, 5%+ decrease)")
plt.xlabel("Hours")
plt.ylabel("Soil Moisture")
plt.xlim(0, 200)  # Set x-axis limits
# plt.legend()
plt.grid()
plt.show()


# In[23]:


# Store recession curve parameters
all_curves = []
for station, curves in recession_curves.items():
    for curve in curves:
        curve_data = {
            'station': station,
            'start_date': curve['start_date'],
            'end_date': curve['end_date'],
            'duration': min(curve['duration'], 200),  # Cap duration at 200 hours for consistency
            'initial_moisture': curve['initial_moisture'],
            'final_moisture': curve['final_moisture'] if curve['duration'] <= 200 else curve['curve'][199],
            'decrease_percent': (curve['initial_moisture'] - curve['final_moisture']) / curve['initial_moisture'] * 100 if curve['duration'] <= 200 else (curve['initial_moisture'] - curve['curve'][199]) / curve['initial_moisture'] * 100
        }
        all_curves.append(curve_data)

curves_df = pd.DataFrame(all_curves)
curves_df.to_csv('data/recession_curve_parameters_usda_ars.csv', index=False)
print(curves_df.head())


# In[ ]:




