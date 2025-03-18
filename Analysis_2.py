#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

def create_performance_map(ax, stations_df, window_hours, cmap):
    ax.set_extent([-125, -66.5, 24, 49], crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.LAND.with_scale('50m'), edgecolor='k', facecolor='#F0F0F0')
    ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='#ADD8E6')
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.5, edgecolor='gray')
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.5, edgecolor='gray')
    
    # Plot all stations
    sc = ax.scatter(
        stations_df['longitude'], stations_df['latitude'],
        c=stations_df['r2'], cmap=cmap, s=100,
        edgecolor='k', linewidth=0.5, alpha=0.8,
        transform=ccrs.PlateCarree(), vmin=0, vmax=1
    )
    
    # Highlight better performing stations
    top_stations = stations_df[stations_df['r2'] > 0.1]
    ax.scatter(
        top_stations['longitude'], top_stations['latitude'],
        s=150, edgecolor='k', linewidth=1, facecolor='none',
        transform=ccrs.PlateCarree()
    )
    
    # Add model performance legend
    handles = [
        Patch(facecolor='#D32F2F', label='R² < 0.3'),
        Patch(facecolor='#FFC107', label='0.3 ≤ R² < 0.6'),
        Patch(facecolor='#4CAF50', label='R² ≥ 0.6'),
        plt.plot([], [], 'o', ms=10, mec='k', mfc='none', label='R² > 0.1')[0]
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=8)
    
    ax.set_title(f'{window_hours}h Prediction Window', fontsize=14, pad=10)
    return sc

# [Previous data loading code remains the same...]

# Create figure
fig = plt.figure(figsize=(20, 15))
gs = GridSpec(2, 2, figure=fig)

# [Previous data processing code remains the same...]

# Create performance maps
for i, window in enumerate(windows):
    ax = fig.add_subplot(gs[i//2, i%2], projection=ccrs.PlateCarree())
    sc = create_performance_map(ax, all_metrics[window], window, cmap)
    if i == 0:
        cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('Model Performance (R²)', fontsize=12)

# Create innovative skill change visualization
ax_skill = fig.add_subplot(gs[1, 1])

# Get stations with R² > 0.1 in 24h predictions
base_stations = all_metrics[24][all_metrics[24]['r2'] > 0.1]['station_id']

# Prepare data for skill change visualization
skill_data = []
for station in base_stations:
    base_r2 = all_metrics[24][all_metrics[24]['station_id'] == station]['r2'].iloc[0]
    r2_48h = all_metrics[48][all_metrics[48]['station_id'] == station]['r2'].iloc[0]
    r2_72h = all_metrics[72][all_metrics[72]['station_id'] == station]['r2'].iloc[0]
    
    skill_data.append({
        'station': station,
        'base_r2': base_r2,
        'change_48h': r2_48h - base_r2,
        'change_72h': r2_72h - base_r2
    })

skill_df = pd.DataFrame(skill_data)

# Calculate skill change statistics
stats_48h = {
    'gained': np.sum(skill_df['change_48h'] > 0),
    'lost': np.sum(skill_df['change_48h'] < 0),
    'mean_gain': skill_df[skill_df['change_48h'] > 0]['change_48h'].mean(),
    'mean_loss': skill_df[skill_df['change_48h'] < 0]['change_48h'].mean()
}

stats_72h = {
    'gained': np.sum(skill_df['change_72h'] > 0),
    'lost': np.sum(skill_df['change_72h'] < 0),
    'mean_gain': skill_df[skill_df['change_72h'] > 0]['change_72h'].mean(),
    'mean_loss': skill_df[skill_df['change_72h'] < 0]['change_72h'].mean()
}

# Add grid
ax_skill.grid(True, linestyle='--', alpha=0.3)

# Create scatter plot with distinctive colors
ax_skill.scatter(skill_df['base_r2'], skill_df['change_48h'], 
                c='#2196F3', s=50, alpha=0.6, label='48h')
ax_skill.scatter(skill_df['base_r2'], skill_df['change_72h'],
                c='#E91E63', s=50, alpha=0.6, label='72h')

ax_skill.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax_skill.set_xlabel('Base Performance (24h R²)', fontsize=12)
ax_skill.set_ylabel('Change in R² Score', fontsize=12)
ax_skill.set_title('Performance Changes vs Base R² for Stations with R² > 0.1', 
                  fontsize=14, pad=10)
ax_skill.legend()

# Add statistics text
stats_text = (
    f"48h Window (blue):\n"
    f"Gained skill: {stats_48h['gained']} stations (mean: +{stats_48h['mean_gain']:.3f})\n"
    f"Lost skill: {stats_48h['lost']} stations (mean: {stats_48h['mean_loss']:.3f})\n\n"
    f"72h Window (pink):\n"
    f"Gained skill: {stats_72h['gained']} stations (mean: +{stats_72h['mean_gain']:.3f})\n"
    f"Lost skill: {stats_72h['lost']} stations (mean: {stats_72h['mean_loss']:.3f})"
)

ax_skill.text(0.02, 0.98, stats_text,
             transform=ax_skill.transAxes,
             verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             fontsize=10)

plt.suptitle('Spatial Performance Analysis Across Prediction Windows', 
            fontsize=16, y=0.95)
plt.tight_layout()

# Save the figure
os.makedirs("Figures", exist_ok=True)
plt.savefig(os.path.join("Figures", "Panel_2.png"), 
            bbox_inches='tight', dpi=300)
plt.close()



# In[ ]:




