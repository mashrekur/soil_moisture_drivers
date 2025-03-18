#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# Optional: Set random seed for reproducibility
np.random.seed(42)

def create_shap_importance_plot(shap_dirs, feature_groups, windows=[24, 48, 72]):
    """Create horizontal bar plot for SHAP values with same style as effect size plot."""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    
    # Color scheme for prediction windows
    colors = ['#2196F3', '#4CAF50', '#E91E63']  # Blue, Green, Pink
    
    # Plot each feature group
    for i, (group_name, features) in enumerate(feature_groups.items()):
        ax = axes[i]
        width = 0.25  # Width of bars
        
        # Split long feature names into two lines
        formatted_features = [f.replace('_', '\n') for f in features]
        
        # Get SHAP values for each window
        indices = np.arange(len(features))
        
        for j, window in enumerate(windows):
            shap_file = os.path.join(shap_dirs[f'{window}h'], 'shap_values.csv')
            if os.path.exists(shap_file):
                shap_df = pd.read_csv(shap_file)
                
                # Calculate mean absolute SHAP values for features
                feature_shaps = []
                for feature in features:
                    if feature in shap_df.columns:
                        shap_value = np.abs(shap_df[feature]).mean()
                        feature_shaps.append(shap_value)
                    else:
                        feature_shaps.append(0)
                
                # Plot bars
                bars = ax.barh(indices + j*width, feature_shaps,
                             width, label=f'{window}h',
                             color=colors[j], alpha=0.8)
                
                # Add value labels
                for bar in bars:
                    width_val = bar.get_width()
                    if width_val > 0:
                        ax.text(width_val + 0.02, bar.get_y() + bar.get_height()/2,
                               f'{width_val:.2f}', va='center', fontsize=8)
        
        # Customize subplot
        ax.set_yticks(indices + width)
        ax.set_yticklabels(formatted_features, fontsize=10)
        ax.set_xlabel("Mean |SHAP value|", fontsize=12)
        ax.set_title(group_name, fontsize=14, pad=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if i == 0:  # Add legend to first subplot
            ax.legend(title='Prediction Window', bbox_to_anchor=(0, 1.1),
                     loc='lower left', fontsize=10)
    
    plt.suptitle('Feature SHAP Values Across Prediction Windows',
                fontsize=16, y=1.05)
    
    # Save figure
    os.makedirs('Figures', exist_ok=True)
    plt.savefig('Figures/Panel_3_SHAP_Importance.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    return fig

def analyze_shap_importance():
    # Define SHAP directories
    shap_dirs = {
        '24h': 'Results/SHAP_Analysis_latest/24h',
        '48h': 'Results/SHAP_Analysis_latest/48h',
        '72h': 'Results/SHAP_Analysis_latest/72h'
    }
    
    # Use same feature groups as effect size analysis
    feature_groups = {
        'Static Physical': [
            'slope', 'cfvo', 'bdod', 'clay_fraction_1', 'clay_fraction_2',
            'sand_fraction_1', 'sand_fraction_2', 'silt_fraction_1', 'silt_fraction_2'
        ],
        'Static Chemical & Hydraulic': [
            'cec', 'nitrogen', 'organic_carbon_1', 'organic_carbon_2',
            'ocs', 'phh2o', 'soc', 'wv0010', 'wv0033', 'wv1500'
        ],
        'Dynamic & Environmental': [
            'temperature', 'precipitation', 'solar_radiation',
            'climate_1_encoded', 'climate_2_encoded', 'landcover_2020_encoded'
        ]
    }
    
    # Create visualization
    fig = create_shap_importance_plot(shap_dirs, feature_groups)
    
    return fig

if __name__ == "__main__":
    # Run SHAP importance analysis
    fig = analyze_shap_importance()
    
    # If in notebook, display the figure
    plt.show()



# In[ ]:





# In[ ]:





# In[1]:


import os
import ast
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_clustered_importance(shap_path, static_path, output_dir):
    """Generate spatial cluster visualization and cluster-specific importance plots"""
    try:
        # Load and convert data
        df = pd.read_csv(shap_path)
        df['dynamic'] = df['dynamic'].apply(ast.literal_eval)
        df['static'] = df['static'].apply(ast.literal_eval)
        
        # Load station coordinates
        with xr.open_dataset(static_path) as ds:
            coords = ds[['station_name', 'latitude', 'longitude']].to_dataframe()
        
        # Merge datasets
        merged = pd.merge(df, coords, left_on='station_id', right_on='station_name')
        
        # Extract dynamic features as separate columns
        dynamic_df = pd.json_normalize(merged['dynamic'])
        required_cols = ['Temperature', 'Precipitation', 'Solar Radiation']
        
        # Validate data
        if not all(col in dynamic_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in dynamic_df.columns]
            raise ValueError(f"Missing required columns in SHAP data: {missing}")
        
        # Cluster stations
        kmeans = KMeans(n_clusters=3, random_state=42)
        merged['cluster'] = kmeans.fit_predict(dynamic_df[required_cols])
        
        # Create cluster labels
        cluster_means = dynamic_df.groupby(merged['cluster']).mean()
        merged['cluster_label'] = merged['cluster'].apply(
            lambda x: cluster_means.loc[x].idxmax()
        )
        
        # Plot spatial distribution
        plt.figure(figsize=(15, 8))
        ax = plt.axes(projection=ccrs.LambertConformal())
        ax.add_feature(cfeature.STATES.with_scale('50m'))
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        
        # Plot stations with cluster colors
        for cluster, group in merged.groupby('cluster_label'):
            ax.scatter(
                group['longitude'], 
                group['latitude'],
                label=cluster,
                s=50,
                transform=ccrs.PlateCarree(),
                edgecolor='w',
                linewidth=0.5
            )
        
        plt.legend(title='Dominant Feature')
        plt.title('Spatial Distribution of Feature Importance Clusters')
        plt.savefig(os.path.join(output_dir, 'spatial_clusters.png'), bbox_inches='tight')
        plt.close()

        # Cluster-specific importance plots
        plt.figure(figsize=(15, 5))
        for idx, (cluster, group) in enumerate(merged.groupby('cluster_label')):
            plt.subplot(1, 3, idx+1)
            cluster_data = pd.json_normalize(group['dynamic'])[required_cols]
            sns.barplot(
                data=cluster_data.melt(),
                x='variable',
                y='value',
                hue='variable',
                estimator=np.mean,
                errorbar=None,
                palette='viridis',
                legend=False
            )
            plt.title(f'{cluster} Cluster (n={len(group)})')
            plt.xticks(rotation=45)
            plt.ylabel('Mean SHAP Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_importances.png'), bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error in plot_clustered_importance: {str(e)}")
        raise

def plot_individual_station_analysis(shap_path, output_dir):
    """Generate detailed visualizations for individual station patterns"""
    try:
        df = pd.read_csv(shap_path)
        df['dynamic'] = df['dynamic'].apply(ast.literal_eval)
        dynamic_df = pd.json_normalize(df['dynamic'])

        plt.figure(figsize=(12, 8))
        ax = plt.gca()  # Get current axes explicitly
        
        # Create scatter plot and store the collection
        scatter = ax.scatter(
            x=dynamic_df['Precipitation'],
            y=dynamic_df['Solar Radiation'],
            c=dynamic_df['Temperature'],
            cmap='coolwarm',
            s=100,
            edgecolor='w'
        )
        
        # Create colorbar using the scatter plot's colormap
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Temperature SHAP Value')
        
        ax.set_xlabel('Precipitation SHAP Value')
        ax.set_ylabel('Solar Radiation SHAP Value')
        ax.set_title('Dynamic Feature Importance Relationships')
        
        plt.savefig(os.path.join(output_dir, 'feature_relationships.png'), bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"Error in plot_individual_station_analysis: {str(e)}")
        raise




if __name__ == "__main__":
    # Configuration
    ANALYSIS_DIR = "Results/SHAP_Analysis_20250213_022351"  # Update with your path
    WINDOW = '24h'
    STATIC_PATH = 'Data/analysis_output/12h_means/static_data.nc'
    
    window_dir = os.path.join(ANALYSIS_DIR, WINDOW)
    shap_path = os.path.join(window_dir, 'shap_values.csv')
    
    # Create output directory for new visualizations
    viz_dir = os.path.join(window_dir, 'advanced_visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate visualizations
    plot_clustered_importance(shap_path, STATIC_PATH, viz_dir)
    plot_individual_station_analysis(shap_path, viz_dir)



# In[ ]:




