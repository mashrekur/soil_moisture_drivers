import pandas as pd
import numpy as np
import xarray as xr
from sklearn.preprocessing import LabelEncoder
import json
import os

def main():
    data_dir = 'Data'
    ismn_data = pd.read_csv(os.path.join(data_dir, 'complete_stations_info.csv'))
    soilgrids_data = pd.read_csv(os.path.join('stations_with_soilgrids_data.csv'))
    landcover_data = pd.read_csv(os.path.join('stations_with_yearly_landcover.csv'))
    slope_data = pd.read_csv(os.path.join('stations_with_slopes_updn.csv'))

    # Merge all data sources
    combined_df = ismn_data.merge(soilgrids_data, on=['latitude', 'longitude'], how='left', suffixes=('', '_soilgrids'))
    combined_df = combined_df.merge(landcover_data, on=['latitude', 'longitude'], how='left', suffixes=('', '_landcover'))
    combined_df = combined_df.merge(slope_data, on=['latitude', 'longitude'], how='left', suffixes=('', '_slope'))

    print("Processing attributes...")

    # Function to split string representations of lists into separate columns for numerical data
    def split_numerical_column(df, column, num_values):
        df[column] = df[column].apply(lambda x: str(x).strip("[]").replace("'", "").split())
        for i in range(num_values):
            df[f'{column}_{i+1}'] = df[column].apply(lambda x: float(x[i]) if len(x) > i else np.nan)
        return df.drop(column, axis=1)

    # Function to split string representations of lists into separate columns for categorical data
    def split_categorical_column(df, column, num_values):
        df[column] = df[column].apply(lambda x: str(x).strip("[]").replace("'", "").split())
        for i in range(num_values):
            df[f'{column}_{i+1}'] = df[column].apply(lambda x: x[i] if len(x) > i else np.nan)
        return df.drop(column, axis=1)

    # List of numerical columns to split
    numerical_columns_to_split = ['saturation', 'clay_fraction', 'organic_carbon', 'sand_fraction', 'silt_fraction']

    # Split the numerical columns
    for column in numerical_columns_to_split:
        combined_df = split_numerical_column(combined_df, column, 2)

    # Split the climate column (categorical)
    combined_df = split_categorical_column(combined_df, 'climate', 2)

    # Define categorical columns
    categorical_columns = ['climate_1', 'climate_2', 'landcover_2020']

    # Define columns to keep for the model
    model_columns = ['station_name', 'latitude', 'longitude', 'slope']
    model_columns.extend([f'{col}_{i}' for col in numerical_columns_to_split for i in [1, 2]])
    model_columns.extend([f'climate_{i}' for i in [1, 2]])
    model_columns.extend(['bdod', 'cec', 'cfvo', 'clay', 'nitrogen', 'ocd', 'ocs', 'phh2o', 'sand', 'silt', 'soc', 'wv0010', 'wv0033', 'wv1500'])
    model_columns.extend(categorical_columns)

    print("Encoding categorical variables...")
    # Encode categorical variables
    for col in categorical_columns:
        if col in combined_df.columns:
            le = LabelEncoder()
            combined_df[f'{col}_encoded'] = le.fit_transform(combined_df[col].astype(str))
            model_columns.append(f'{col}_encoded')
            
            # Store encoding mapping in a dictionary
            encoding_map = dict(zip(le.classes_.tolist(), le.transform(le.classes_).tolist()))
            combined_df[f'{col}_encoding'] = combined_df[col].map(encoding_map)

    # Keep only the relevant columns
    model_df = combined_df[model_columns]

    # Check for duplicate columns
    duplicate_columns = model_df.columns[model_df.columns.duplicated()].tolist()
    if duplicate_columns:
        print(f"Warning: Duplicate columns found: {duplicate_columns}")
        # Remove duplicate columns
        model_df = model_df.loc[:, ~model_df.columns.duplicated()]

    print("Creating xarray Dataset...")
    # Create xarray Dataset
    ds = xr.Dataset.from_dataframe(model_df.set_index('station_name'))

    # Prepare metadata
    metadata = {
        'categorical_columns': categorical_columns,
        'encoding_info': {
            col: dict(zip(combined_df[col].unique().tolist(), combined_df[f'{col}_encoded'].unique().tolist()))
            for col in categorical_columns if col in combined_df.columns
        },
        'model_columns': model_df.columns.tolist()  # Update model_columns to reflect any changes
    }

    print("Saving data and metadata...")
    # Save data and metadata
    nc_path = os.path.join(data_dir, 'static_attributes_model_v3.nc')
    json_path = os.path.join(data_dir, 'static_attributes_model_metadata_v3.json')

    ds.to_netcdf(nc_path)
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Model-ready static attributes saved to '{nc_path}'")
    print(f"Metadata saved to '{json_path}'")

    print("Processing completed.")

if __name__ == "__main__":
    main()



