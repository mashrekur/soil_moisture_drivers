#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Set paths
model_dirs = {
    '24h': "unique_stations",
    '48h': "unique_stations_48h",
    '72h': "unique_stations_72h"
}
figures_dir = "Figures"
os.makedirs(figures_dir, exist_ok=True)

# Load 24h metrics to determine top stations
metrics_path = os.path.join(model_dirs['24h'], "all_stations_metrics.csv")
metrics_df = pd.read_csv(metrics_path, header=0, index_col=0)
metrics_df = metrics_df.reset_index().rename(columns={"index": "station_id"})
metrics_df['station_id'] = metrics_df['station_id'].astype(str)

# Sort stations by 24h R² and select top 25
top_25_stations = metrics_df.sort_values(by="r2", ascending=False).head(25)

# Create figure with custom legend area
fig = plt.figure(figsize=(50, 30))
gs = fig.add_gridspec(6, 5, height_ratios=[0.2] + [1]*5)  # First row for legend

# Create legend axes
legend_ax = fig.add_subplot(gs[0, :])
legend_ax.axis('off')

# Define colors and styles
colors = {
    'actual': '#1A237E',  # Dark blue
    '24h': '#4CAF50',     # Green
    '48h': '#C20078',     # Orange
    '72h': '#A9561E'      # Red
}

# Create custom legend
legend_elements = [
    Line2D([0], [0], color=colors['actual'], linewidth=4, label='Actual'),
    Line2D([0], [0], color=colors['24h'], linewidth=4, label='24h Prediction'),
    Line2D([0], [0], color=colors['48h'], linewidth=4, label='48h Prediction'),
    Line2D([0], [0], color=colors['72h'], linewidth=4, label='72h Prediction')
]
legend_ax.legend(handles=legend_elements, loc='center', ncol=4, fontsize=25)

# Create subplots for stations
axes = []
for i in range(5):
    for j in range(5):
        ax = fig.add_subplot(gs[i+1, j])
        axes.append(ax)

# Plot each station
for i, (_, row) in enumerate(top_25_stations.iterrows()):
    station_id = row['station_id']
    ax = axes[i]
    
    # Plot predictions for each window
    for window, color in colors.items():
        if window == 'actual':
            continue
            
        try:
            # Load predictions
            predictions_path = os.path.join(model_dirs[window], station_id, "predictions.csv")
            predictions_df = pd.read_csv(predictions_path)
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
            
            # Plot actual values only once
            if window == '24h':
                ax.plot(predictions_df['timestamp'], predictions_df['actual'],
                       color=colors['actual'], linewidth=2, label='Actual')
            
            # Plot predictions
            ax.plot(predictions_df['timestamp'], predictions_df['predicted'],
                   color=color, linewidth=1.5, alpha=0.5)
            
        except Exception as e:
            print(f"Error plotting {window} predictions for station {station_id}: {str(e)}")
    
    # Customize subplot
    ax.set_title(f"{station_id}\n24h R²: {row['r2']:.3f}", fontsize=18)
    ax.tick_params(axis='both', labelsize=15)
    ax.grid(True, linestyle='--', alpha=0.8, color='gray')
    
    # Format x-axis
    ax.set_xlim(predictions_df['timestamp'].min(), predictions_df['timestamp'].max())
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    
    # Only add y-label for leftmost plots
    if i % 5 == 0:
        ax.set_ylabel('Soil Moisture', fontsize=15)
    
    # Only add x-label for bottom plots
    if i >= 20:
        ax.set_xlabel('Time', fontsize=15)

# Add overall title
fig.suptitle('Multi-window Prediction Comparison for Top 25 Stations\nRanked by 24h Test R² (01/01/2019-12/31/2019)',
            fontsize=30, y=1.02)

# Adjust layout
plt.tight_layout()

# Save figure
output_path = os.path.join(figures_dir, "Panel_1.png")
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()



# In[ ]:




