# Drivers of Soil Moisture Dynamics over Continental United States

Welcome to the GitHub repository for the research paper "Drivers of Soil Moisture Dynamics over Continental United States", submitted to Water Resources Research. This codebase supports a novel study exploring the complex drivers of soil moisture dynamics across the Continental United States (CONUS) using a spatially-aware Vision Transformer (ViT) architecture. Our approach integrates local and regional influences, leveraging data from 360 International Soil Moisture Network (ISMN) stations alongside comprehensive environmental datasets—ERA5 climate reanalysis, USGS elevation products, MODIS land cover, and SoilGrids soil characteristics. The resulting model achieves systematic improvements in prediction skill over 24, 48, and 72-hour forecast horizons, revealing key insights into soil-environment interactions through dual-method feature importance analysis (Cohen's d and SHAP values).
This repository contains a comprehensive set of Python scripts organized into three main categories:

**Data Acquisition & Preprocessing Scripts (0.1–1.5):** These scripts handle the extraction, processing, and alignment of diverse datasets critical to the study. They acquire static attributes (e.g., soil properties, slope, land cover) and dynamic variables (e.g., temperature, precipitation, solar radiation) from various sources, preprocess soil moisture time series, and align data for 2015–2019 across common stations. Outputs are saved as NetCDF files, with visualizations to explore spatial and temporal patterns.


**Core Model & Training Scripts (dataloader_individual.py, ViT_LSTM_Individual.py, training_individual.py):** These scripts implement the core of our spatially-aware ViT model (EnhancedSpatiotemporalModel) and its training pipeline. The dataloader prepares station-specific sequences, the model integrates spatial and temporal features using a Transformer encoder and k-nearest neighbors, and the training script optimizes the model per station with quantile loss, saving predictions and metrics.


**Model Analysis Scripts (Analysis_1.py–Analysis_5.py):** These scripts analyze model performance and feature importance across prediction windows. They generate visualizations such as time series comparisons (Analysis_1), spatial performance maps (Analysis_2), effect size and correlation plots (Analysis_3), SHAP-based clustering (Analysis_4), and a comprehensive SHAP analysis panel (Analysis_5), highlighting drivers like precipitation, solar radiation, and clay content.

Detailed documentation for each script is provided below, including functionality, inputs/outputs, and dependencies.


## Data Acquisition & Preprocessing Scripts:

0.1_extract_ismn_static_attributes: This script extracts metadata from ISMN station files in a specified directory, including coordinates and static variables like soil properties (e.g., saturation, clay fraction).

0.2_extract_ismn_soil_moisture_prototype.py: This script processes soil moisture time series data from USDA-ARS stations, extracting and analyzing recession curves (periods of decreasing moisture). It generates time series plots, histograms with fitted distributions, and recession curve visualizations, saving statistical summaries and curve parameters to CSV files.

0.3_extract_epq_3dem_slope.py: This script calculates topographic slopes for weather stations using elevation data from an online service, processes station data from a CSV file, and generates visualizations. It creates a 2D colormap and a 3D contour map showing slope variations across station locations, along with a histogram of slope values, and saves the results to files.

0.4_climatic_data.py: This script downloads hourly ERA5 reanalysis data (temperature and precipitation) for the contiguous United States  using the CDS API. It processes the data to generate histograms, monthly averages, and spatial maps for July averages, and creates animations of temperature and precipitation variations over time, both for a week in 2002 and daily averages across all years.

0.5_soilgrids.py: This script retrieves soil property data (e.g., clay, sand, organic carbon) from the SoilGrids API for weather station locations listed in a CSV file. It merges this data with station information, saves the results to a CSV, and generates visualizations including point maps over the contiguous United States for each soil parameter, using different colormaps to represent variations.

0.6_get_landcover_data.py: This script uses Google Earth Engine to extract MODIS land cover data for weather station locations in the contiguous United States from 2015 to the latest available year. It updates a station DataFrame with yearly land cover values, saves the results to a CSV, and creates interactive Folium maps displaying land cover layers for selected years, with a dropdown widget for year selection.

0.7_combine_static_attr.py: This script combines static attribute data from multiple sources (ISMN, SoilGrids, land cover, and slope) into a single DataFrame, processes numerical and categorical variables by splitting and encoding them, and prepares the data for modeling. It creates an xarray Dataset, saves it as a NetCDF file, and stores metadata (e.g., encoding mappings) in a JSON file.

0.8_combine_target_attr.py: This script downloads ERA5 reanalysis data (temperature, precipitation, and solar radiation) for a bounding box around weather station locations from 2000 to 2023 using the CDS API. It processes the raw data by converting units (e.g., precipitation to mm, temperature to °C, solar radiation to W/m²), saves both raw and processed datasets as NetCDF files, and includes logging for tracking progress and errors.
1.0_visualize_dynamic_input_targets.py: This script loads processed ERA5 data and station information from files, extracts specific variables (e.g., temperature) at station locations, and generates visualizations. It creates maps showing variable values at stations over the contiguous United States for a specific time and an average over the year, plus a time series plot for a selected station, using Cartopy for mapping.

1.1_common_stations_data.py: This script integrates static, dynamic, and target data for a set of common weather stations over the period 2015–2019. It processes static attributes from a NetCDF file, extracts ERA5 dynamic data (e.g., temperature, precipitation) for station locations, and aligns ISMN soil moisture target data to a consistent hourly time index, saving each dataset as a NetCDF file in a combined output directory

1.2_check_processed_datasets.py: This script preprocesses soil moisture data from a combined ISMN dataset by setting negative values to zero and filling gaps with interpolated values, saving progress with checkpoints. It loads static and dynamic datasets, aligns them by common stations, saves filtered versions as NetCDF files, and provides detailed analysis of dataset structures, station counts, and alignment statistics, with optional visualizations of preprocessed data.

1.3_data_alignment.py: This script loads target (soil moisture), static, and dynamic (ERA5) datasets, aligns them by matching station names, and handles duplicates in the static data. It creates filtered datasets with consistent station sets, saves them as NetCDF files, and generates visualizations (time series and bar plots) for a randomly selected station’s soil moisture, static attributes, and temperature data.

1.4_check_aligned_data.py: This script aligns target (soil moisture), static, and dynamic (ERA5) datasets by matching stations, handling duplicates in the static data by keeping entries with the most non-null values, and adding coordinates to the dynamic dataset. It saves aligned datasets as NetCDF files, reshapes the target data for consistency, and generates visualizations including maps of dynamic variables (e.g., temperature), static variables (e.g., slope), and target soil moisture across selected time points over the contiguous United States, with detailed logging and analysis of data structure and NaN patterns.

1.5_create_station_map.py: This script generates a detailed map of ISMN stations used in a study, displaying station locations across the contiguous United States with state boundaries, coastlines, and abbreviations, alongside a list of station names. It also creates a grid of maps visualizing static (e.g., slope, soil properties) and dynamic (e.g., temperature) variables from the latest clipped dataset, using high-resolution Cartopy features and a consistent layout for all variables.

## Core Model & Training Scripts:

dataloader_individual.py: This script defines the IndividualStationDataset class and a helper function get_station_loaders for loading data specific to individual stations in a PyTorch-based soil moisture prediction model. The dataset class loads static (e.g., soil properties), dynamic (e.g., weather), and target (soil moisture) data from NetCDF files using xarray, filters by station ID and optional time range, and prepares sequences of specified length (seq_len) and prediction horizon (pred_len). It normalizes latitude and longitude coordinates and converts timestamps to UNIX time, returning data as PyTorch tensors. The helper function creates training and testing DataLoaders with configurable batch sizes and shuffling.

ViT_LSTM_Individual.py: This script defines the EnhancedSpatiotemporalModel class, a PyTorch neural network combining spatial and temporal processing for soil moisture prediction. It integrates static (e.g., soil properties) and dynamic (e.g., weather) features using a spatial encoder (MLP with GELU activation), a static feature processor incorporating k-nearest neighbors, and a dynamic feature embedder. A Transformer encoder processes temporal sequences, followed by a prediction head with learnable scaling parameters. The model uses a BallTree for spatial neighbor queries based on latitude/longitude (in radians), enhancing predictions with spatial context from neighboring stations.

training_individual.py: This script implements the training pipeline for the EnhancedSpatiotemporalModel on a per-station basis. It uses the IndividualStationDataset and DataLoaders to train the model with a quantile loss function (quantiles 0.1, 0.5, 0.9) and AdamW optimizer. For each station, it trains over a specified number of epochs, evaluates performance on a test set using RMSE and R² metrics, saves the best model based on R², and generates prediction plots and metrics files. The script processes all stations listed in the static data, saving results in a unique_stations directory, and includes memory management (garbage collection, CUDA cache clearing) for stability.


## Model Analysis Scripts:

Analysis_1.py: This script visualizes soil moisture predictions for the top 25 stations based on 24-hour R² performance. It loads metrics and prediction data for 24h, 48h, and 72h forecast windows, creating a 5x5 grid of time series plots comparing actual and predicted values. Each subplot includes station-specific R² values, with a custom legend and detailed formatting, saved as a high-resolution figure.

Analysis_2.py: This script  analyzes model performance (R²) across 24h, 48h, and 72h prediction windows for ISMN stations. It creates a figure with three CONUS maps showing spatial R² distribution for each window, using Cartopy for high-resolution features, and a scatter plot comparing performance changes (48h and 72h vs. 24h baseline) for stations with R² > 0.1, including statistics on skill gains/losses, saved as a high-resolution figure.

Analysis_3.py: This script, likely named Analysis_3.py, performs a comprehensive analysis of feature importance and relationships across 24h, 48h, and 72h prediction windows for soil moisture modeling. It calculates effect sizes (Cohen's d) for static and dynamic features, visualizes them in a comparative bar plot grouped by feature type, and analyzes trends to identify dominant and lag features. Additionally, it computes Spearman correlations at the station level, saving them for each window, and generates chord diagrams and heatmaps to illustrate feature relationships, with Panel 4 combining these visualizations into a multi-panel figure with enhanced styling and tighter spacing.

Analysis_4.py: This script conducts an advanced analysis of SHAP (SHapley Additive exPlanations) values for soil moisture prediction models across 24h, 48h, and 72h windows. It generates a horizontal bar plot of mean absolute SHAP values for feature groups, styled consistently with prior effect size plots, and performs K-means clustering (3 clusters) on dynamic feature SHAP values (temperature, precipitation, solar radiation). The script produces spatial maps showing cluster distribution over CONUS using Cartopy, bar plots of cluster-specific feature importance, and a scatter plot exploring relationships between dynamic feature SHAP values, all saved as high-resolution figures in a dedicated visualization directory.

Analysis_5.py: This script, likely named Analysis_5.py, creates a comprehensive visualization (Panel 5) of SHAP values for soil moisture prediction across 24h, 48h, and 72h windows. It parses SHAP results, normalizing values for consistency, and generates a multi-panel figure: (A) a bar plot of top static feature impacts across windows, (B) a bar plot of dynamic feature (temperature, precipitation, solar radiation) impacts, and (C) a horizontal bar plot of relative SHAP value changes from 24h to 72h for both static and dynamic features. The script includes robust error handling, dynamic feature detection, and professional styling, saving the output as a high-resolution figure.



























































