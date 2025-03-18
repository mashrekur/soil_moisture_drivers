#!/usr/bin/env python
# coding: utf-8

# In[11]:


import ee
import geemap
import pandas as pd
import ipywidgets as widgets
from ipyleaflet import WidgetControl


# In[4]:


# Initialize Earth Engine
ee.Authenticate()
ee.Initialize()


# In[10]:


# Define CONUS boundary
conus = ee.FeatureCollection("TIGER/2018/States").filter(ee.Filter.inList('NAME', ['Alaska', 'Hawaii']).Not())

# Load MODIS Land Cover collection
modis_lc = ee.ImageCollection("MODIS/006/MCD12Q1").select('LC_Type1')

# Find the latest available year
latest_year = ee.Date(modis_lc.aggregate_max('system:time_start')).get('year').getInfo()
print(f"Latest available year: {latest_year}")

# Load the stations dataframe
stations_df = pd.read_csv('stations_info.csv')

# Convert stations to Earth Engine feature collection
stations_ee = ee.FeatureCollection(
    [ee.Feature(ee.Geometry.Point([row['longitude'], row['latitude']]), {'id': str(i)}) 
     for i, row in stations_df.iterrows()]
)

# Define the years you want to process
years = range(2015, latest_year + 1)

for year in years:
    print(f"Processing year {year}")
    
    # Filter the collection for the specific year
    landcover = modis_lc.filter(ee.Filter.date(f'{year}-01-01', f'{year}-12-31')).first()
    
    # Clip to CONUS
    landcover = landcover.clip(conus)
    
    # Extract land cover values for all stations
    landcover_values = landcover.reduceRegions(
        collection=stations_ee,
        reducer=ee.Reducer.first(),
        scale=500
    )
    
    # Get the results as a list
    results = landcover_values.getInfo()['features']
    
    # Update the dataframe
    for result in results:
        station_id = int(result['properties']['id'])
        lc_value = result['properties'].get('first')
        if lc_value is not None:
            stations_df.at[station_id, f'landcover_{year}'] = lc_value

# Save the updated dataframe
stations_df.to_csv('stations_with_yearly_landcover.csv', index=False)


# In[14]:


import folium

# Create a map centered on the US
m = folium.Map(location=[40, -98], zoom_start=4)

# Display the map
m


# In[15]:


# Function to add Earth Engine layer to folium map
def add_ee_layer(self, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)

# Add EE drawing method to folium.
folium.Map.add_ee_layer = add_ee_layer

# Create a folium map
m = folium.Map(location=[40, -98], zoom_start=4)

# Load MODIS Land Cover collection
modis_lc = ee.ImageCollection("MODIS/006/MCD12Q1").select('LC_Type1')

# Get the most recent image
recent_lc = modis_lc.sort('system:time_start', False).first()

# Define visualization parameters
vis_params = {
    'min': 1,
    'max': 17,
    'palette': [
        '1c0dff', '05450a', '086a10', '54a708', '78d203', '009900', 'c6b044',
        'dcd159', 'dade48', 'fbff13', 'b6ff05', '27ff87', 'c24f44', 'a5a5a5',
        'ff6d4c', '69fff8', 'f9ffa4', '1c0dff'
    ]
}

# Add the land cover layer to the map
m.add_ee_layer(recent_lc, vis_params, 'MODIS Land Cover')

# Add layer control
folium.LayerControl().add_to(m)

# Display the map
m


# In[17]:


# Define CONUS boundary
conus = ee.FeatureCollection("TIGER/2018/States").filter(ee.Filter.inList('NAME', ['Alaska', 'Hawaii']).Not())

# Load MODIS Land Cover collection
modis_lc = ee.ImageCollection("MODIS/006/MCD12Q1").select('LC_Type1')

# Find the latest available year
latest_year = ee.Date(modis_lc.aggregate_max('system:time_start')).get('year').getInfo()

# Function to add Earth Engine layer to folium map
def add_ee_layer(self, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)

# Add EE drawing method to folium.
folium.Map.add_ee_layer = add_ee_layer

# Create a folium map
m = folium.Map(location=[40, -98], zoom_start=4)

# Define visualization parameters
vis_params = {
    'min': 1,
    'max': 17,
    'palette': [
        '1c0dff', '05450a', '086a10', '54a708', '78d203', '009900', 'c6b044',
        'dcd159', 'dade48', 'fbff13', 'b6ff05', '27ff87', 'c24f44', 'a5a5a5',
        'ff6d4c', '69fff8', 'f9ffa4', '1c0dff'
    ]
}

# Function to get land cover for a specific year
def get_landcover(year):
    return modis_lc.filter(ee.Filter.date(f'{year}-01-01', f'{year}-12-31')).first().clip(conus)

# Add land cover layers for multiple years
years_to_display = [2015, 2017, latest_year]
for year in years_to_display:
    lc = get_landcover(year)
    m.add_ee_layer(lc, vis_params, f'Land Cover {year}')

# Add CONUS boundary
conus_style = {'fillColor': '#00000000', 'color': '#000000'}
folium.GeoJson(
    data=conus.getInfo(),
    style_function=lambda x: conus_style,
    name='CONUS Boundary'
).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Display the map
display(m)

# Create a dropdown widget for year selection
year_dropdown = widgets.Dropdown(
    options=[(str(year), year) for year in years_to_display],
    value=years_to_display[0],
    description='Year:',
)

# Display the dropdown
display(year_dropdown)


# In[ ]:




