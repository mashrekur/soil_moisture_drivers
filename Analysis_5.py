#!/usr/bin/env python
# coding: utf-8

# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import ast
from matplotlib.gridspec import GridSpec

def parse_shap_results(file_path):
    """Parse SHAP results file with dictionary strings."""
    df = pd.read_csv(file_path)
    
    # Create new dataframe for results
    results = []
    
    for _, row in df.iterrows():
        # Parse dictionaries
        static_dict = ast.literal_eval(row['static'])
        dynamic_dict = ast.literal_eval(row['dynamic'])
        
        # Combine into single row
        result_row = {
            'station_id': row['station_id'],
            **static_dict,
            **dynamic_dict
        }
        results.append(result_row)
    
    return pd.DataFrame(results)

def create_panel_5(base_dir):
    """Create comprehensive visualization of SHAP analysis results."""
    plt.style.use('default')
    
    # Set up figure
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2])
    
    # Load and process data for all windows
    windows = ['24h', '48h', '72h']
    all_results = {}
    
    for window in windows:
        file_path = f"Results/{base_dir}/{window}/shap_values.csv"
        all_results[window] = parse_shap_results(file_path)
        
        # Debugging: Print columns to verify feature names
        print(f"Columns in {window} results:")
        print(all_results[window].columns)
    
    # Define dynamic features (update these based on the actual column names)
    dynamic_features = ['Temperature', 'Precipitation', 'Solar Radiation']  # Adjust as needed
    
    # 1. Top left: Temporal evolution of dynamic feature importance
    ax1 = fig.add_subplot(gs[0, 0])
    
    temporal_importance = {window: {
        feature: results[feature].mean() 
        for feature in dynamic_features
    } for window, results in all_results.items()}
    
    temporal_df = pd.DataFrame(temporal_importance).T
    
    # Plot with custom colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # warm red, teal, light blue
    temporal_df.plot(kind='bar', ax=ax1, width=0.8, color=colors)
    
    ax1.set_title('Dynamic Feature Importance Across Prediction Windows', fontsize=14)
    ax1.set_xlabel('Prediction Window', fontsize=12)
    ax1.set_ylabel('Mean |SHAP Value|', fontsize=12)
    ax1.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Top right: Distribution of feature importance
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create violin plots for dynamic features
    violin_data = []
    for window in windows:
        for feature in dynamic_features:
            violin_data.extend([{
                'Window': window,
                'Feature': feature.capitalize(),
                'SHAP Value': value
            } for value in all_results[window][feature]])
    
    violin_df = pd.DataFrame(violin_data)
    
    sns.violinplot(data=violin_df, x='Window', y='SHAP Value', 
                  hue='Feature', ax=ax2, palette=colors)
    
    ax2.set_title('Distribution of Feature Importance', fontsize=14)
    ax2.set_xlabel('Prediction Window', fontsize=12)
    ax2.set_ylabel('|SHAP Value|', fontsize=12)
    
    # 3. Bottom: Static feature importance comparison
    ax3 = fig.add_subplot(gs[1, :])
    
    # Get static features (excluding dynamic ones and station_id)
    static_features = [col for col in all_results['24h'].columns 
                      if col not in ['station_id'] + dynamic_features]
    
    # Calculate mean importance for static features
    static_importance = pd.DataFrame({
        window: all_results[window][static_features].mean() 
        for window in windows
    })
    
    # Sort features by average importance
    static_importance['mean'] = static_importance.mean(axis=1)
    static_importance = static_importance.sort_values('mean', ascending=True)
    static_importance = static_importance.drop('mean', axis=1)
    
    # Plot static feature importance
    static_importance.plot(kind='barh', ax=ax3, width=0.8, 
                         color=['#FF9F1C', '#E71D36', '#2EC4B6'])
    
    ax3.set_title('Static Feature Importance Comparison', fontsize=14)
    ax3.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax3.set_ylabel('Feature', fontsize=12)
    ax3.legend(title='Prediction Window')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig('Figures/panel_5_shap_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    print("\nDynamic Feature Importance Summary:")
    print(temporal_df)
    print("\nTop 5 Most Important Static Features:")
    print(static_importance.mean(axis=1).sort_values(ascending=False).head())
    
    return static_importance, temporal_df



# Run the visualization
static_importance, temporal_importance = create_panel_5('SHAP_Analysis_Latest')



# In[49]:


import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Helper Functions
def safe_literal_eval(s):
    """Safely convert string representations of dictionaries"""
    try:
        return ast.literal_eval(str(s))
    except (ValueError, SyntaxError):
        return {}

def normalize_shap_values(shap_df):
    """Normalize SHAP values to [0, 1] range for better visualization"""
    # Ensure all values are numeric
    shap_df = shap_df.apply(pd.to_numeric, errors='coerce')
    return shap_df / shap_df.abs().max()

def parse_shap_results(file_path):
    """Robust parser with error handling and data validation"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"SHAP results file not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")
    
    # Validate required columns
    required_cols = {'station_id', 'static', 'dynamic'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert dictionary columns
    for col in ['static', 'dynamic']:
        df[col] = df[col].apply(safe_literal_eval)
    
    # Explode dictionary columns and ensure numeric values
    static_df = pd.json_normalize(df['static']).apply(pd.to_numeric, errors='coerce')  # Remove prefix
    dynamic_df = pd.json_normalize(df['dynamic']).add_prefix('dynamic_').apply(pd.to_numeric, errors='coerce')
    
    # Combine with station IDs
    combined_df = pd.concat([
        df['station_id'],
        static_df,
        dynamic_df
    ], axis=1)
    
    return combined_df

# Main Visualization Function
def create_panel_5(results_dir):
    """Create enhanced SHAP analysis visualization with normalized values and relative changes"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })
    
    # Set up figure with professional layout
    fig = plt.figure(figsize=(20, 18))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5],
                width_ratios=[1, 1], hspace=0.6, wspace=0.4)  # Increased spacing
    
    # Color scheme
    dynamic_colors = ['#2E86AB', '#A23B72', '#F18F01']  # For dynamic features (Panel B)
    static_colors = ['#4CAF50', '#FFC107', '#9C27B0']   # For static features (Panel A)

    # Load and process data
    windows = ['24h', '48h', '72h']
    all_results = {}
    
    for window in windows:
        file_path = os.path.join(results_dir, window, 'shap_values.csv')
        all_results[window] = parse_shap_results(file_path)
        all_results[window] = normalize_shap_values(all_results[window])  # Normalize SHAP values

    # Dynamically detect features
    sample_window = list(all_results.values())[0]
    dynamic_features = [col.replace('dynamic_', '') 
                       for col in sample_window.columns 
                       if col.startswith('dynamic_')]
    static_features = [col for col in sample_window.columns 
                      if col not in ['station_id'] and not col.startswith('dynamic_')]  # Exclude dynamic features

    # 1. Top Static Feature Impacts (Panel A)
    ax1 = fig.add_subplot(gs[0, :])  # Span full width
    static_importance = pd.concat(
        [all_results[window][static_features].mean()
         for window in windows], axis=1)
    static_importance.columns = windows
    
    # Include more static features by lowering the threshold
    top_features = static_importance.mean(axis=1).nlargest(30).index  # Increase to 30 features
    static_importance = static_importance.loc[top_features]
    
    # Plot 3 bars per feature (24h, 48h, 72h)
    x = np.arange(len(top_features))  # Feature labels
    width = 0.25  # Width of bars
    
    for i, window in enumerate(windows):
        ax1.bar(x + i * width, static_importance[window], width=width,
                color=static_colors[i], label=window, edgecolor='black')
    
    ax1.set_title('A) Top Static Feature Impacts', y=1.02)
    ax1.set_xlabel('Static Features')
    ax1.set_ylabel('Normalized SHAP Value')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(top_features, rotation=45, ha='right')
    ax1.legend(title='Prediction Window', loc='upper right')  # Legend inside the plot
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Dynamic Feature Impact (Panel B)
    ax2 = fig.add_subplot(gs[1, 0])
    temporal_data = {window: [np.mean(results[f'dynamic_{feat}']) 
                    for feat in dynamic_features]
                    for window, results in all_results.items()}
    
    df_temporal = pd.DataFrame(temporal_data, index=dynamic_features).T
    df_temporal.plot(kind='bar', ax=ax2, color=dynamic_colors, width=0.8, edgecolor='white')
    
    ax2.set_title('B) Dynamic Feature Impact', y=1.02)  # Renamed
    ax2.set_xlabel('Prediction Window')
    ax2.set_ylabel('Normalized SHAP Value')
    ax2.legend(title='Dynamic Features', loc='upper right')  # Legend inside the plot
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # 3. Relative Change in Importance (Panel C)
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Calculate relative change for static and dynamic features
    relative_change_static = (static_importance['72h'] - static_importance['24h']) / static_importance['24h']
    relative_change_dynamic = pd.DataFrame({
        feat: (all_results['72h'][f'dynamic_{feat}'].mean() - all_results['24h'][f'dynamic_{feat}'].mean()) / 
              all_results['24h'][f'dynamic_{feat}'].mean()
        for feat in dynamic_features
    }, index=['Relative Change'])
    
    # Combine static and dynamic relative changes
    relative_change = pd.concat([relative_change_static, relative_change_dynamic.T])
    relative_change = relative_change.sort_values(by=0, ascending=False)
    
    # Plot relative changes
    relative_change.plot(kind='barh', ax=ax3, color=static_colors[0], width=0.8, alpha=0.7, legend=False)
    
    ax3.set_title('C) Relative Change in Importance (24h to 72h)', y=1.02)
    ax3.set_xlabel('Relative Change')
    ax3.set_ylabel('')
    ax3.grid(axis='x', linestyle='--', alpha=0.7)

    # Final adjustments
    plt.suptitle("SHAP Feature Impact Analysis", y=0.98,  # Renamed
                fontsize=16, weight='bold')
    
    # Save figure
    os.makedirs('Figures', exist_ok=True)
    plt.savefig('Figures/panel_5_shap_analysis.png', dpi=300, 
               bbox_inches='tight', facecolor='white')
    plt.close()

    return static_importance, relative_change

# Main Execution
def main():
    try:
        # Define the base directory containing SHAP results
        results_dir = 'Results/SHAP_Analysis_Latest'
        
        # Run visualization
        static_imp, relative_change = create_panel_5(results_dir)
        
        # Print summary statistics
        print("Top Static Feature Importance Summary:")
        print(static_imp)
        print("\nRelative Change in Importance (24h to 72h):")
        print(relative_change)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Troubleshooting Tips:")
        print("1. Verify SHAP results files exist in: Results/SHAP_Analysis_Latest/{window}/shap_values.csv")
        print("2. Ensure CSV files contain 'station_id', 'static', and 'dynamic' columns")
        print("3. Check that 'static' and 'dynamic' columns contain valid dictionary strings")
        print("4. Confirm feature names are consistent across analysis and visualization")

if __name__ == "__main__":
    main()



# In[ ]:




