#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import xarray as xr

def calculate_effect_size(feature_values, performance_values):
    median_performance = np.median(performance_values)
    high_perform = feature_values[performance_values > median_performance]
    low_perform = feature_values[performance_values <= median_performance]
    
    n1, n2 = len(high_perform), len(low_perform)
    var1, var2 = high_perform.var(), low_perform.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    effect_size = np.abs((high_perform.mean() - low_perform.mean()) / pooled_std)
    return effect_size

def create_comparative_effect_size_plot(effect_sizes_dict, feature_groups):
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
        
        # Get effect sizes for each window
        indices = np.arange(len(features))
        
        for j, (window, effects) in enumerate(effect_sizes_dict.items()):
            # Extract effects for current features
            feature_effects = [effects.get(feature, 0) for feature in features]
            
            # Plot bars
            bars = ax.barh(indices + j*width, feature_effects,
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
        ax.set_yticklabels(formatted_features, fontsize=10)  # Use formatted feature names
        ax.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
        ax.set_title(group_name, fontsize=14, pad=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if i == 0:  # Add legend to first subplot
            ax.legend(title='Prediction Window', bbox_to_anchor=(0, 1.1),
                     loc='lower left', fontsize=10)
    
    plt.suptitle('Feature Importance Analysis Across Prediction Windows',
                fontsize=16, y=1.05)
    
    # Save figure
    os.makedirs('Figures', exist_ok=True)
    plt.savefig('Figures/Panel_3_Feature_Importance.png',
                dpi=300, bbox_inches='tight')
    plt.close()

def analyze_multi_window_importance(config):
    windows = [24, 48, 72]
    all_effects = {}
    
    for window in windows:
        metrics_path = f"unique_stations{'_' + str(window) + 'h' if window > 24 else ''}/all_stations_metrics.csv"
        metrics_df = pd.read_csv(metrics_path, header=0, index_col=0)
        
        # Handle different column names
        metrics_df = metrics_df.reset_index()
        if window == 24:
            # Handle empty header for 24h window
            metrics_df = metrics_df.rename(columns={metrics_df.columns[0]: "station_id"})
        else:
            # Handle 'station' column for 48/72h
            metrics_df = metrics_df.rename(columns={"station": "station_id"})

        window_effects = {}
        
        # Analyze static features
        with xr.open_dataset(config['static_path']) as static_data:
            static_features = static_data.drop_vars(['latitude', 'longitude']).to_dataframe()
            static_features = static_features.reset_index()
            
            analysis_df = pd.merge(
                static_features,
                metrics_df,
                left_on='station_name',
                right_on='station_id'
            )
            
            for column in static_features.columns:
                if column not in ['station_name', 'station_id']:
                    effect = calculate_effect_size(analysis_df[column], analysis_df['r2'])
                    if not np.isnan(effect) and effect > 0:
                        window_effects[column] = effect
        
        # Analyze dynamic features
        with xr.open_dataset(config['dynamic_path']) as dynamic_data:
            for feature in ['temperature', 'precipitation', 'solar_radiation']:
                station_effects = []
                valid_stations = []
                
                for station in metrics_df['station_id']:
                    if station in dynamic_data['station'].values:
                        station_data = dynamic_data[feature].sel(station=station)
                        temporal_var = station_data.std().item() / station_data.mean().item()
                        station_effects.append(temporal_var)
                        valid_stations.append(station)
                
                if station_effects:
                    effect = calculate_effect_size(
                        np.array(station_effects),
                        metrics_df[metrics_df['station_id'].isin(valid_stations)]['r2']
                    )
                    if not np.isnan(effect) and effect > 0:
                        window_effects[feature] = effect
        
        all_effects[window] = pd.Series(window_effects)
    
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
    
    create_comparative_effect_size_plot(all_effects, feature_groups)

    pd.DataFrame(all_effects).to_csv('effect_size_results.csv')
    
    return all_effects

def analyze_trends(effects):
    # Convert effects to a DataFrame for easier analysis
    trends = {}
    for window, effects_series in effects.items():
        trends[f'{window}h'] = effects_series.sort_values(ascending=False)
    
    trends_df = pd.DataFrame(trends)
    
    # Print top features for each window
    print("Top Features by Prediction Window:")
    print(trends_df.head(10))
    
    # Analyze trends across windows
    print("\nTrends Across Prediction Windows:")
    
    # Check if certain features become more dominant over longer windows
    dominant_features = {}
    for feature in trends_df.index:
        window_effects = trends_df.loc[feature]
        if window_effects.idxmax() == '72h':  # Feature is most important in 72h window
            dominant_features[feature] = window_effects
    
    if dominant_features:
        print("\nFeatures that become more dominant over longer prediction windows:")
        for feature, effects in dominant_features.items():
            print(f"{feature}: {effects.to_dict()}")
    else:
        print("No features show increasing dominance over longer prediction windows.")
    
    # Check for lag variables (e.g., dynamic features like temperature, precipitation)
    lag_candidates = ['temperature', 'precipitation', 'solar_radiation']
    lag_features = {}
    for feature in lag_candidates:
        if feature in trends_df.index:
            window_effects = trends_df.loc[feature]
            if window_effects.idxmax() != '24h':  # Lag variables may peak in 48h/72h
                lag_features[feature] = window_effects
    
    if lag_features:
        print("\nPotential lag variables (dynamic features peaking in 48h/72h):")
        for feature, effects in lag_features.items():
            print(f"{feature}: {effects.to_dict()}")
    else:
        print("No clear lag variables identified.")
    
    # Check for static feature dominance
    static_features = [
        'slope', 'cfvo', 'bdod', 'clay_fraction_1', 'clay_fraction_2',
        'sand_fraction_1', 'sand_fraction_2', 'silt_fraction_1', 'silt_fraction_2',
        'cec', 'nitrogen', 'organic_carbon_1', 'organic_carbon_2',
        'ocs', 'phh2o', 'soc', 'wv0010', 'wv0033', 'wv1500'
    ]
    static_dominance = {}
    for feature in static_features:
        if feature in trends_df.index:
            window_effects = trends_df.loc[feature]
            if window_effects.idxmax() == '72h':  # Static features peaking in 72h
                static_dominance[feature] = window_effects
    
    if static_dominance:
        print("\nStatic features that dominate in longer prediction windows:")
        for feature, effects in static_dominance.items():
            print(f"{feature}: {effects.to_dict()}")
    else:
        print("No static features show clear dominance in longer prediction windows.")
    
    return trends_df

if __name__ == "__main__":
    config = {
        'static_path': 'Data/analysis_output/12h_means/static_data.nc',
        'dynamic_path': 'Data/analysis_output/12h_means/dynamic_data_12h.nc'
    }
    
    # Run feature importance analysis
    effects = analyze_multi_window_importance(config)
    
    # Analyze trends in the results
    trends_df = analyze_trends(effects)



# In[17]:


import pandas as pd
import numpy as np
import xarray as xr
import os
from scipy import stats

def analyze_multi_window_importance(config):
    """Enhanced analysis that saves station-level data"""
    windows = [24, 48, 72]
    all_effects = {}
    
    # Create output directory
    os.makedirs('station_level_data', exist_ok=True)
    
    for window in windows:
        print(f"\nProcessing {window}h window...")
        metrics_path = f"unique_stations{'_' + str(window) + 'h' if window > 24 else ''}/all_stations_metrics.csv"
        metrics_df = pd.read_csv(metrics_path, header=0, index_col=0)
        
        # Handle different column names
        metrics_df = metrics_df.reset_index()
        if window == 24:
            metrics_df = metrics_df.rename(columns={metrics_df.columns[0]: "station_id"})
        else:
            metrics_df = metrics_df.rename(columns={"station": "station_id"})

        window_effects = {}
        
        # Create station-level dataframe
        station_data = {'station_id': [], 'r2': [], 'feature': [], 'value': []}
        
        # Analyze static features
        with xr.open_dataset(config['static_path']) as static_data:
            static_features = static_data.drop_vars(['latitude', 'longitude']).to_dataframe()
            static_features = static_features.reset_index()
            
            analysis_df = pd.merge(
                static_features,
                metrics_df,
                left_on='station_name',
                right_on='station_id'
            )
            
            # Save merged static data
            analysis_df.to_csv(f'station_level_data/static_features_{window}h.csv')
            
            # Calculate effect sizes and store station-level data
            for column in static_features.columns:
                if column not in ['station_name', 'station_id']:
                    # Store station-level data
                    for _, row in analysis_df.iterrows():
                        station_data['station_id'].append(row['station_id'])
                        station_data['r2'].append(row['r2'])
                        station_data['feature'].append(column)
                        station_data['value'].append(row[column])
                    
                    # Calculate effect size
                    effect = calculate_effect_size(analysis_df[column], analysis_df['r2'])
                    if not np.isnan(effect) and effect > 0:
                        window_effects[column] = effect
        
        # Analyze dynamic features
        with xr.open_dataset(config['dynamic_path']) as dynamic_data:
            dynamic_stats = []
            
            for feature in ['temperature', 'precipitation', 'solar_radiation']:
                station_effects = []
                valid_stations = []
                
                for station in metrics_df['station_id']:
                    if station in dynamic_data['station'].values:
                        station_data = dynamic_data[feature].sel(station=station)
                        temporal_var = station_data.std().item() / station_data.mean().item()
                        station_effects.append(temporal_var)
                        valid_stations.append(station)
                        
                        # Store station-level dynamic feature statistics
                        dynamic_stats.append({
                            'station_id': station,
                            'feature': feature,
                            'mean': station_data.mean().item(),
                            'std': station_data.std().item(),
                            'cv': temporal_var
                        })
                
                if station_effects:
                    # Calculate effect size
                    effect = calculate_effect_size(
                        np.array(station_effects),
                        metrics_df[metrics_df['station_id'].isin(valid_stations)]['r2']
                    )
                    if not np.isnan(effect) and effect > 0:
                        window_effects[feature] = effect
            
            # Save dynamic feature statistics
            pd.DataFrame(dynamic_stats).to_csv(
                f'station_level_data/dynamic_features_{window}h.csv'
            )
        
        # Save station-level data
        pd.DataFrame(station_data).to_csv(
            f'station_level_data/all_features_{window}h.csv'
        )
        
        # Use string keys for window names (e.g., '24h', '48h', '72h')
        all_effects[f'{window}h'] = pd.Series(window_effects)
    
    # Save aggregated effect sizes
    pd.DataFrame(all_effects).to_csv('effect_size_results.csv')
    
    return all_effects, 'station_level_data'


# Save feature correlations for chord plot
def calculate_feature_correlations(data_dir, window='24h'):
    """Calculate feature correlations using station-level data"""
    # Load static features
    static_df = pd.read_csv(f'{data_dir}/static_features_{window}.csv')
    dynamic_df = pd.read_csv(f'{data_dir}/dynamic_features_{window}.csv')
    
    # Prepare correlation dataframe
    features = [col for col in static_df.columns 
               if col not in ['station_id', 'station_name', 'r2']]
    features.extend(['temperature', 'precipitation', 'solar_radiation'])
    
    correlations = pd.DataFrame(index=features, columns=features)
    
    # Calculate correlations
    for f1 in features:
        for f2 in features:
            if f1 != f2:
                if f1 in static_df.columns and f2 in static_df.columns:
                    # Static-Static correlation
                    corr = stats.spearmanr(static_df[f1], static_df[f2])[0]
                elif f1 in ['temperature', 'precipitation', 'solar_radiation'] and \
                     f2 in ['temperature', 'precipitation', 'solar_radiation']:
                    # Dynamic-Dynamic correlation
                    vals1 = dynamic_df[dynamic_df['feature'] == f1]['cv']
                    vals2 = dynamic_df[dynamic_df['feature'] == f2]['cv']
                    corr = stats.spearmanr(vals1, vals2)[0]
                else:
                    # Static-Dynamic correlation
                    if f1 in static_df.columns:
                        static_feat, dynamic_feat = f1, f2
                    else:
                        static_feat, dynamic_feat = f2, f1
                    
                    dynamic_vals = dynamic_df[dynamic_df['feature'] == dynamic_feat]['cv']
                    static_vals = static_df[static_feat]
                    corr = stats.spearmanr(static_vals, dynamic_vals)[0]
            else:
                corr = 1.0
                
            correlations.loc[f1, f2] = corr
    
    # Save correlations
    correlations.to_csv(f'station_level_data/feature_correlations_{window}.csv')
    return correlations

import networkx as nx
import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import opts, dim
import colorcet as cc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

def create_chord_diagram(data_dir, window='24h', min_correlation=0.3):
    """
    Create a proper circular chord diagram showing feature relationships.
    """
    # Load data and clean it
    correlations = pd.read_csv(f'{data_dir}/feature_correlations_{window}.csv', index_col=0)
    
    # Remove unwanted columns and rows
    unwanted_cols = ['Unnamed: 0', 'rmse']
    correlations = correlations.drop(unwanted_cols, errors='ignore')
    correlations = correlations.drop(unwanted_cols, axis=1, errors='ignore')
    
    # Feature groups for coloring
    feature_groups = {
        'Dynamic Weather': [
            'temperature', 'precipitation', 'solar_radiation'
        ],
        'Physical Properties': [
            'slope', 'cfvo', 'bdod', 
            'clay_fraction_1', 'clay_fraction_2',
            'sand_fraction_1', 'sand_fraction_2', 
            'silt_fraction_1', 'silt_fraction_2',
            'saturation_1', 'saturation_2',
            'clay', 'sand', 'silt'
        ],
        'Chemical Properties': [
            'cec', 'nitrogen', 
            'organic_carbon_1', 'organic_carbon_2',
            'ocs', 'ocd', 'phh2o', 'soc'
        ],
        'Hydraulic Properties': [
            'wv0010', 'wv0033', 'wv1500'
        ],
        'Environmental': [
            'climate_1_encoded', 'climate_2_encoded', 
            'landcover_2020_encoded'
        ]
    }
    
    # Function to find feature group
    def get_feature_group(feature):
        for group, features in feature_groups.items():
            if feature in features:
                return group
        print(f"Warning: Feature '{feature}' not found in any group, adding to Physical Properties")
        feature_groups['Physical Properties'].append(feature)
        return 'Physical Properties'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_aspect('equal')
    
    # Set up the colors with a more distinctive palette
    colors = plt.cm.Dark2(np.linspace(0, 1, len(feature_groups)))
    group_colors = dict(zip(feature_groups.keys(), colors))
    
    # Calculate positions
    features = list(correlations.index)
    n_features = len(features)
    
    # Sort features by group to keep related features together
    features_by_group = []
    for group in feature_groups:
        group_features = [f for f in features if get_feature_group(f) == group]
        features_by_group.extend(group_features)
    
    # Calculate angles for each feature with spacing between groups
    angles = np.linspace(0, 2*np.pi, len(features_by_group), endpoint=False)
    
    # Radius for the main circle
    R = 1.0
    
    # Create the outer circle segments and labels
    for idx, feature in enumerate(features_by_group):
        # Get feature group
        group = get_feature_group(feature)
        color = group_colors[group]
        
        # Calculate arc angles
        theta1 = angles[idx] - 0.1
        theta2 = angles[idx] + 0.1
        
        # Draw arc
        arc = patches.Arc((0, 0), R*2, R*2, theta1=np.degrees(theta1), 
                         theta2=np.degrees(theta2), color=color, lw=10)
        ax.add_patch(arc)
        
        # Add label
        label_angle = angles[idx]
        label_x = (R + 0.2) * np.cos(label_angle)
        label_y = (R + 0.2) * np.sin(label_angle)
        
        # Adjust text alignment based on position
        ha = 'left' if -np.pi/2 <= label_angle <= np.pi/2 else 'right'
        rotation = np.degrees(label_angle)
        if rotation > 90:
            rotation -= 180
        if rotation < -90:
            rotation += 180
            
        plt.text(label_x, label_y, feature, 
                rotation=rotation,
                ha=ha, va='center',
                fontsize=8, fontweight='bold')
    
    # Draw the chords
    max_correlation = correlations.abs().max().max()  # For normalizing widths
    
    for i, source in enumerate(features_by_group):
        for j, target in enumerate(features_by_group):
            if i < j:
                correlation = abs(correlations.loc[source, target])
                if correlation >= min_correlation:
                    # Calculate control points for Bezier curve
                    start_angle = angles[i]
                    end_angle = angles[j]
                    
                    # Start and end points
                    start = (R * np.cos(start_angle), R * np.sin(start_angle))
                    end = (R * np.cos(end_angle), R * np.sin(end_angle))
                    
                    # Control points
                    # Adjust radius based on correlation strength
                    radius_factor = 0.3 + 0.7 * (correlation / max_correlation)
                    control1 = (radius_factor * np.cos(start_angle), radius_factor * np.sin(start_angle))
                    control2 = (radius_factor * np.cos(end_angle), radius_factor * np.sin(end_angle))
                    
                    # Create Bezier curve
                    verts = [start, control1, control2, end]
                    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
                    path = Path(verts, codes)
                    
                    # Draw the chord with width proportional to correlation
                    width = correlation / max_correlation * 5  # Normalize width
                    patch = patches.PathPatch(path, facecolor='none', 
                                           edgecolor='gray',
                                           alpha=0.6,
                                           linewidth=width*3)
                    ax.add_patch(patch)
    
    # Add title with correlation range
    plt.title(f'Feature Relationships ({window} Prediction Window)\n' +
              f'Correlation Range: {min_correlation:.2f} - {max_correlation:.2f}', 
              fontsize=16, pad=20)
    
    # Set limits and remove axes
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    
    # Add legend for feature groups
    legend_elements = [patches.Patch(facecolor=color, label=group)
                      for group, color in group_colors.items()]
    ax.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0, 0),
             fontsize=10)
    
    # Save figure
    plt.savefig(f'Figures/chord_diagram_{window}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'Figures/chord_diagram_{window}.svg', bbox_inches='tight')
    
    return fig




def get_unique_feature_color(feature):
    """Assign color based on feature type"""
    if feature in ['temperature', 'precipitation', 'solar_radiation']:
        return '#1f77b4'  # Dynamic features
    elif any(x in feature for x in ['climate', 'landcover']):
        return '#2ca02c'  # Environmental features
    elif any(x in feature for x in ['clay', 'sand', 'silt']):
        return '#ff7f0e'  # Physical features
    else:
        return '#d62728'  # Other features



def analyze_feature_relationships(data_dir='station_level_data'):
    """Run comprehensive feature relationship analysis"""
    windows = ['24h', '48h', '72h']
    results = {}
    
    for window in windows:
        print(f"\nAnalyzing {window} prediction window...")
        
        # Create chord diagram
        chord_fig = create_chord_diagram(data_dir, window)
        
        # Create correlation heatmap
        heatmap_fig = create_correlation_heatmap(data_dir, window)
        
        results[window] = {
            'chord_diagram': chord_fig,
            'correlation_heatmap': heatmap_fig
        }
        
        plt.close()  # Clean up figures
    
    return results



def create_correlation_heatmap(data_dir, window='24h'):
    """Create supplementary heatmap of feature correlations"""
    correlations = pd.read_csv(f'{data_dir}/feature_correlations_{window}.csv', index_col=0)
    effect_sizes = pd.read_csv('effect_size_results.csv', index_col=0)
    
    # Filter features
    significant_features = effect_sizes[str(window)].dropna()
    significant_features = significant_features[significant_features > 0.1].index
    correlations = correlations.loc[significant_features, significant_features]
    
    # Create heatmap
    plt.figure(figsize=(15, 12))
    heatmap = plt.imshow(correlations, cmap='RdBu_r', aspect='auto')
    plt.colorbar(heatmap)
    
    # Add feature labels
    plt.xticks(range(len(correlations.columns)), correlations.columns, rotation=45, ha='right')
    plt.yticks(range(len(correlations.index)), correlations.index)
    
    plt.title(f'Feature Correlations ({window} Prediction Window)')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'Figures/correlation_heatmap_{window}.png', dpi=300, bbox_inches='tight')
    return plt.gcf()

def analyze_feature_relationships(data_dir='station_level_data'):
    """Run comprehensive feature relationship analysis"""
    windows = ['24h', '48h', '72h']
    results = {}
    
    for window in windows:
        print(f"\nAnalyzing {window} prediction window...")
        
        # Create chord diagram
        chord_fig = create_chord_diagram(data_dir, window)
        
        # Create correlation heatmap
        heatmap_fig = create_correlation_heatmap(data_dir, window)
        
        results[window] = {
            'chord_diagram': chord_fig,
            'correlation_heatmap': heatmap_fig
        }
        
        plt.close()  # Clean up figures
    
    return results

# Configuration
config = {
    'static_path': 'Data/analysis_output/12h_means/static_data.nc',
    'dynamic_path': 'Data/analysis_output/12h_means/dynamic_data_12h.nc'
}

# 1. Run enhanced analysis
all_effects, data_dir = analyze_multi_window_importance(config)

# 2. Calculate correlations for each window
for window in ['24h', '48h', '72h']:
    correlations = calculate_feature_correlations(data_dir, window)
    print(f"Calculated correlations for {window} window")

# 3. Now create chord diagrams
results = analyze_feature_relationships(data_dir)





# In[19]:


def debug_correlations(data_dir):
    """
    Debug correlation values across different prediction windows.
    """
    windows = ['24h', '48h', '72h']
    
    for window in windows:
        print(f"\nAnalyzing {window} prediction window:")
        
        # Load correlation data
        corr_file = f'{data_dir}/feature_correlations_{window}.csv'
        correlations = pd.read_csv(corr_file, index_col=0)
        
        # Remove unwanted columns
        unwanted_cols = ['Unnamed: 0', 'rmse']
        correlations = correlations.drop(unwanted_cols, errors='ignore')
        correlations = correlations.drop(unwanted_cols, axis=1, errors='ignore')
        
        # Get dynamic variables correlations
        dynamic_vars = ['temperature', 'precipitation', 'solar_radiation']
        dynamic_corr = correlations.loc[dynamic_vars, :]
        
        # Print statistics for dynamic variables
        print("\nCorrelation statistics for dynamic variables:")
        for var in dynamic_vars:
            # Get correlations excluding self-correlation
            var_corr = dynamic_corr.loc[var, :]
            var_corr = var_corr[var_corr.index != var]
            
            print(f"\n{var}:")
            print(f"Mean correlation: {var_corr.mean():.3f}")
            print(f"Max correlation: {var_corr.max():.3f}")
            print(f"Number of correlations > 0.3: {(var_corr > 0.3).sum()}")
            
            # Print top 5 strongest correlations
            print("Top 5 correlations:")
            top_corr = var_corr.sort_values(ascending=False)[:5]
            for feat, corr in top_corr.items():
                print(f"{feat}: {corr:.3f}")

    return dynamic_corr

# Function to reload and verify data
def verify_data_loading(data_dir, window):
    """
    Verify how the data is being loaded and processed.
    """
    print(f"\nVerifying data for {window} window:")
    
    try:
        # Load original data files
        static_data = xr.open_dataset(f"{data_dir}/static_data.nc")
        dynamic_data = xr.open_dataset(f"{data_dir}/dynamic_data.nc")
        target_data = xr.open_dataset(f"{data_dir}/target_data.nc")
        
        print("\nData shapes:")
        print(f"Static data: {static_data.dims}")
        print(f"Dynamic data: {dynamic_data.dims}")
        print(f"Target data: {target_data.dims}")
        
        # Load correlation file
        corr_file = f'{data_dir}/feature_correlations_{window}.csv'
        correlations = pd.read_csv(corr_file, index_col=0)
        
        print("\nCorrelation matrix shape:", correlations.shape)
        print("Number of features:", len(correlations.columns))
        
        return True
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False
# Run debug analysis
dynamic_corr = debug_correlations('station_level_data')

# Verify data loading for each window
for window in ['24h', '48h', '72h']:
    verify_data_loading('station_level_data', window)



# In[27]:


def calculate_correlations(data_dir, window='24h'):
    """
    Calculate feature correlations with proper handling of temporal dependencies.
    
    Args:
        data_dir: Directory containing the data files
        window: Prediction window (24h, 48h, or 72h)
    """
    # Load data
    target_data = pd.read_csv(f'{data_dir}/soil_moisture_data.csv')
    dynamic_data = pd.read_csv(f'{data_dir}/dynamic_data.csv')
    static_data = pd.read_csv(f'{data_dir}/static_data.csv')
    
    # Convert window to hours
    hours = int(window.replace('h', ''))
    
    # For dynamic variables, calculate rolling statistics over the prediction window
    dynamic_stats = {}
    for var in ['temperature', 'precipitation', 'solar_radiation']:
        # Calculate mean over the prediction window
        dynamic_stats[f'{var}_mean'] = dynamic_data[var].rolling(window=hours).mean()
        # Calculate variance over the prediction window
        dynamic_stats[f'{var}_var'] = dynamic_data[var].rolling(window=hours).var()
        # Calculate max over the prediction window
        dynamic_stats[f'{var}_max'] = dynamic_data[var].rolling(window=hours).max()
    
    # Combine dynamic stats with static features
    features = pd.concat([pd.DataFrame(dynamic_stats), static_data], axis=1)
    
    # Calculate target variable for the prediction window
    target = target_data['soil_moisture'].shift(-hours)
    
    # Calculate correlations
    correlations = pd.DataFrame(index=features.columns, columns=features.columns)
    for col1 in features.columns:
        for col2 in features.columns:
            # Use Spearman correlation to capture non-linear relationships
            correlation = spearmanr(features[col1].dropna(), features[col2].dropna())[0]
            correlations.loc[col1, col2] = correlation
    
    # Save correlations
    correlations.to_csv(f'{data_dir}/feature_correlations_{window}.csv')
    
    return correlations

def analyze_temporal_correlations(data_dir):
    """
    Analyze how correlations change across prediction windows.
    """
    windows = ['24h', '48h', '72h']
    results = {}
    
    for window in windows:
        # Calculate correlations for each window
        correlations = calculate_correlations(data_dir, window)
        results[window] = correlations
        
        # Print summary statistics
        print(f"\nAnalysis for {window} prediction window:")
        dynamic_vars = ['temperature', 'precipitation', 'solar_radiation']
        
        for var in dynamic_vars:
            var_corrs = correlations.loc[var, :].abs()
            print(f"\n{var}:")
            print(f"Mean correlation: {var_corrs.mean():.3f}")
            print(f"Max correlation: {var_corrs.max():.3f}")
            print(f"Number of strong correlations (>0.3): {(var_corrs > 0.3).sum()}")
            print("\nTop 5 correlations:")
            print(var_corrs.nlargest(5))
    
    return results
# Import required libraries
import pandas as pd
import numpy as np
from scipy.stats import spearmanr



# First let's check what's in your correlation files
data_dir = 'station_level_data'

# Load and print the correlations that already exist
for window in ['24h', '48h', '72h']:
    print(f"\nAnalyzing {window} prediction window:")
    corr_file = f'{data_dir}/feature_correlations_{window}.csv'
    correlations = pd.read_csv(corr_file, index_col=0)
    
    # Remove unwanted columns
    unwanted_cols = ['Unnamed: 0', 'rmse']
    correlations = correlations.drop(unwanted_cols, errors='ignore')
    correlations = correlations.drop(unwanted_cols, axis=1, errors='ignore')
    
    # Focus on dynamic variables
    dynamic_vars = ['temperature', 'precipitation', 'solar_radiation']
    
    for var in dynamic_vars:
        var_corr = correlations.loc[var, :].abs()  # Get absolute correlations
        significant_corr = var_corr[var_corr > 0.3]  # Get correlations > 0.3
        
        print(f"\n{var}:")
        print(f"Top correlations:")
        print(significant_corr.sort_values(ascending=False).head())





# In[29]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, to_rgba

def create_chord_diagram(data_dir, window='24h', min_correlation=0.3):
    """
    Create chord diagram emphasizing temporal relationships.
    """
    # Load correlations
    correlations = pd.read_csv(f'{data_dir}/feature_correlations_{window}.csv', index_col=0)
    
    # Remove unwanted columns
    unwanted_cols = ['Unnamed: 0', 'rmse']
    correlations = correlations.drop(unwanted_cols, errors='ignore')
    correlations = correlations.drop(unwanted_cols, axis=1, errors='ignore')
    
    # Feature groups with proper categorization
    feature_groups = {
        'Dynamic Weather': [
            'temperature', 'precipitation', 'solar_radiation'
        ],
        'Physical Properties': [
            'slope', 'cfvo', 'bdod', 
            'clay_fraction_1', 'clay_fraction_2',
            'sand_fraction_1', 'sand_fraction_2', 
            'silt_fraction_1', 'silt_fraction_2',
            'saturation_1', 'saturation_2',
            'clay', 'sand', 'silt'
        ],
        'Chemical Properties': [
            'cec', 'nitrogen', 
            'organic_carbon_1', 'organic_carbon_2',
            'ocs', 'ocd', 'phh2o', 'soc'
        ],
        'Hydraulic Properties': [
            'wv0010', 'wv0033', 'wv1500'
        ],
        'Environmental': [
            'climate_1_encoded', 'climate_2_encoded', 
            'landcover_2020_encoded'
        ]
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_aspect('equal')
    
    # Define color scheme
    group_colors = {
        'Dynamic Weather': '#E41A1C',    # Red
        'Physical Properties': '#377EB8', # Blue
        'Chemical Properties': '#4DAF4A', # Green
        'Hydraulic Properties': '#984EA3',# Purple
        'Environmental': '#FF7F00'        # Orange
    }
    
    # Get features and sort by group
    features = []
    for group in feature_groups:
        features.extend([f for f in correlations.index if f in feature_groups[group]])
    
    # Calculate angles with gaps between groups
    n_features = len(features)
    angles = np.linspace(0, 2*np.pi, n_features, endpoint=False)
    
    # Create the outer circle segments and labels
    R = 1.0  # Base radius
    current_angle = 0
    feature_angles = {}
    
    for group_name, group_features in feature_groups.items():
        group_size = sum(1 for f in features if f in group_features)
        if group_size == 0:
            continue
            
        group_angle = 2 * np.pi * group_size / n_features
        
        # Draw group arc
        theta1 = np.degrees(current_angle)
        theta2 = np.degrees(current_angle + group_angle)
        arc = patches.Arc((0, 0), R*2.2, R*2.2, theta1=theta1, theta2=theta2,
                         color=group_colors[group_name], lw=10, alpha=0.3)
        ax.add_patch(arc)
        
        # Add group features
        for feature in [f for f in features if f in group_features]:
            angle = current_angle + angles[features.index(feature)]
            feature_angles[feature] = angle
            
            # Calculate label position
            label_r = R + 0.3
            x = label_r * np.cos(angle)
            y = label_r * np.sin(angle)
            
            # Rotate text
            rotation = np.degrees(angle)
            if rotation > 90 and rotation < 270:
                rotation += 180
                ha = 'right'
            else:
                ha = 'left'
                
            plt.text(x, y, feature, rotation=rotation,
                    ha=ha, va='center', fontsize=8,
                    color=group_colors[group_name],
                    fontweight='bold')
        
        current_angle += group_angle
    
    # Draw the chords
    drawn_pairs = set()
    max_correlation = correlations.abs().max().max()
    
    for i, source in enumerate(features):
        for j, target in enumerate(features):
            if i < j and (source, target) not in drawn_pairs:
                correlation = abs(correlations.loc[source, target])
                if correlation >= min_correlation:
                    # Calculate chord width and color
                    width = (correlation / max_correlation) * 5
                    
                    # Get group colors for gradient
                    source_group = next(g for g, f in feature_groups.items() if source in f)
                    target_group = next(g for g, f in feature_groups.items() if target in f)
                    
                    # Create gradient color
                    source_color = to_rgba(group_colors[source_group])
                    target_color = to_rgba(group_colors[target_group])
                    
                    # Draw chord with gradient color
                    verts = [
                        (R * np.cos(feature_angles[source]), R * np.sin(feature_angles[source])),
                        (0, 0),
                        (R * np.cos(feature_angles[target]), R * np.sin(feature_angles[target]))
                    ]
                    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                    
                    path = Path(verts, codes)
                    patch = patches.PathPatch(path, facecolor='none',
                                           edgecolor='gray',
                                           alpha=min(0.8, correlation),
                                           linewidth=width)
                    ax.add_patch(patch)
                    drawn_pairs.add((source, target))
    
    # Add title with window information
    plt.title(f'Feature Relationships\n{window} Prediction Window\n(correlations ≥ {min_correlation})',
             fontsize=16, pad=20)
    
    # Set limits and remove axes
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.axis('off')
    
    # Add legend
    legend_elements = [patches.Patch(facecolor=color, label=group)
                      for group, color in group_colors.items()]
    ax.legend(handles=legend_elements, loc='center', fontsize=10)
    
    # Save figures
    plt.savefig(f'Figures/chord_diagram_{window}_temporal.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'Figures/chord_diagram_{window}_temporal.svg', bbox_inches='tight')
    
    return fig
# Create diagrams for all windows
for window in ['24h', '48h', '72h']:
    print(f"\nCreating chord diagram for {window} window...")
    fig = create_chord_diagram('station_level_data', window)
    plt.close(fig)




# In[ ]:





# In[13]:


import pandas as pd

data_dir = 'station_level_data'
df_48h = pd.read_csv(f'{data_dir}/feature_correlations_48h.csv', index_col=0)
df_72h = pd.read_csv(f'{data_dir}/feature_correlations_72h.csv', index_col=0)
print("Are CSV files identical?", df_48h.equals(df_72h))
print("48h head:\n", df_48h.head())
print("72h head:\n", df_72h.head())


# In[19]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.gridspec import GridSpec

def create_panel_4(data_dir='station_level_data', min_correlation=0.1):
    """
    Create Panel 4 with tighter vertical spacing between subplots.
    """
    plt.rcParams.update({'font.size': 12.5})
    SMALL_SIZE = 12.5
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 30
    XTICK_SIZE = 10

    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=XTICK_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)
    
    # Reduced figure height for tighter layout
    fig = plt.figure(figsize=(25, 30))  # Changed from (25, 40) to (25, 30)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], width_ratios=[3, 2])
    gs.update(top=0.96, hspace=0.05)  # Adjusted top to 0.96, hspace to 0.05
    
    colors = ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0",
             "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors)
    
    feature_groups = {
        'Dynamic Weather': ['temperature', 'precipitation', 'solar_radiation'],
        'Physical Properties': ['slope', 'cfvo', 'bdod', 'clay_fraction_1', 'clay_fraction_2',
                              'sand_fraction_1', 'sand_fraction_2', 'silt_fraction_1', 'silt_fraction_2',
                              'saturation_1', 'saturation_2', 'clay', 'sand', 'silt'],
        'Chemical Properties': ['cec', 'nitrogen', 'organic_carbon_1', 'organic_carbon_2',
                              'ocs', 'ocd', 'phh2o', 'soc'],
        'Hydraulic Properties': ['wv0010', 'wv0033', 'wv1500'],
        'Environmental': ['climate_1_encoded', 'climate_2_encoded', 'landcover_2020_encoded']
    }
    
    group_colors = {
        'Dynamic Weather': '#E41A1C',
        'Physical Properties': '#377EB8',
        'Chemical Properties': '#4DAF4A',
        'Hydraulic Properties': '#984EA3',
        'Environmental': '#FF7F00'
    }
    
    all_features = [feat for group in feature_groups.values() for feat in group]
    correlations_dict = {}
    
    for i, window in enumerate(['24h', '48h', '72h']):
        correlations = pd.read_csv(f'{data_dir}/feature_correlations_{window}.csv', index_col=0)
        common_features = correlations.index.intersection(correlations.columns).intersection(all_features)
        correlations = correlations.loc[common_features, common_features]
        correlations_dict[window] = correlations
        
        ax_heat = fig.add_subplot(gs[i, 0])
        sns.heatmap(correlations.abs(), cmap=custom_cmap, vmin=0, vmax=1,
                   square=True, ax=ax_heat,
                   cbar_kws={'label': 'Correlation Magnitude', 'shrink': 0.5, 'aspect': 5},
                   linewidths=0.5)
        
        ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha='right')
        ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0)
        plt.setp(ax_heat.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        # Adjusted heatmap size to fill more vertical space
        box = ax_heat.get_position()
        ax_heat.set_position([box.x0, box.y0 + box.height * 0.1, box.width * 0.8, box.height * 0.8])  # Changed from 0.6 to 0.8
        ax_heat.set_title(f'{window} Prediction Window', pad=15)  # Reduced pad from 20 to 15
        
        ax_chord = fig.add_subplot(gs[i, 1])
        create_chord_subplot(correlations, feature_groups, group_colors, ax_chord, 
                           window, min_correlation)
    
    print("Verifying differences between prediction windows:")
    for window in ['24h', '48h', '72h']:
        mean_corr = correlations_dict[window].abs().mean().mean()
        print(f"{window} - Mean absolute correlation: {mean_corr:.6f}")
    
    print("\nAre 48h and 72h identical after filtering?")
    identical = correlations_dict['48h'].equals(correlations_dict['72h'])
    print(f"48h vs 72h identical: {identical}")
    if not identical:
        diff = correlations_dict['48h'] - correlations_dict['72h']
        print("Differences (48h - 72h):")
        print(diff[diff != 0].dropna(how='all').to_string())
        mean_diff = diff.abs().mean().mean()
        print(f"Mean absolute difference between 48h and 72h: {mean_diff:.6f}")
    else:
        print("No differences found in filtered data.")
    
    fig.suptitle('Feature Relationships Across Prediction Windows', 
                fontsize=BIGGER_SIZE, y=0.98)
    plt.tight_layout()
    plt.savefig('Figures/Panel_4_relationships.png', dpi=300, bbox_inches='tight')
    plt.savefig('Figures/Panel_4_relationships.svg', bbox_inches='tight')
    return fig

def create_chord_subplot(correlations, feature_groups, group_colors, ax, window, min_correlation):
    ax.set_aspect('equal')
    features = []
    for group in feature_groups:
        features.extend([f for f in correlations.index if f in feature_groups[group]])
    
    n_features = len(features)
    total_angle = 2 * np.pi * 0.9
    angles = np.linspace(0, total_angle, n_features, endpoint=False)
    
    R = 1.0
    current_angle = 0
    feature_angles = {}
    
    for group_name, group_features in feature_groups.items():
        group_size = sum(1 for f in features if f in group_features)
        if group_size == 0:
            continue
        group_angle = total_angle * group_size / n_features
        current_angle += 0.1
        theta1 = np.degrees(current_angle)
        theta2 = np.degrees(current_angle + group_angle)
        arc = patches.Arc((0, 0), R*2.2, R*2.2, theta1=theta1, theta2=theta2,
                         color=group_colors[group_name], lw=10, alpha=0.3)
        ax.add_patch(arc)
        
        for idx, feature in enumerate([f for f in features if f in group_features]):
            angle = current_angle + (group_angle * idx / (group_size + 1))
            feature_angles[feature] = angle
            label_r = R + 0.4
            x = label_r * np.cos(angle)
            y = label_r * np.sin(angle)
            rotation = np.degrees(angle)
            if rotation > 90 and rotation < 270:
                rotation += 180
                ha = 'right'
            else:
                ha = 'left'
            plt.text(x, y, feature, rotation=rotation,
                    ha=ha, va='center',
                    fontsize=12.5,
                    color=group_colors[group_name],
                    fontweight='bold')
        current_angle += group_angle
    
    drawn_pairs = set()
    max_correlation = correlations.abs().max().max()
    
    for i, source in enumerate(features):
        for j, target in enumerate(features):
            if i < j and (source, target) not in drawn_pairs:
                correlation = abs(correlations.loc[source, target])
                if correlation >= min_correlation:
                    width = (correlation / max_correlation) * 5
                    source_group = next(g for g, f in feature_groups.items() if source in f)
                    target_group = next(g for g, f in feature_groups.items() if target in f)
                    verts = [
                        (R * np.cos(feature_angles[source]), R * np.sin(feature_angles[source])),
                        (0, 0),
                        (R * np.cos(feature_angles[target]), R * np.sin(feature_angles[target]))
                    ]
                    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                    path = Path(verts, codes)
                    patch = patches.PathPatch(path, facecolor='none',
                                           edgecolor='gray',
                                           alpha=min(0.8, correlation),
                                           linewidth=width*3)
                    ax.add_patch(patch)
                    drawn_pairs.add((source, target))
    
    legend_elements = [patches.Patch(facecolor=color, label=group)
                      for group, color in group_colors.items()]
    ax.legend(handles=legend_elements, 
             loc='center left',
             bbox_to_anchor=(1.05, 0.5))
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.axis('off')
    ax.set_title(f'{window} Prediction Window', pad=15)  # Reduced pad from 20 to 15

if __name__ == "__main__":
    panel_4 = create_panel_4()


# In[ ]:




