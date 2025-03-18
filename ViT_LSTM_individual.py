import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import BallTree

class EnhancedSpatiotemporalModel(nn.Module):
    def __init__(self, static_dim, dynamic_dim, num_heads, num_layers, 
                 hidden_dim, ff_dim, dropout, prediction_length, k_neighbors=5):
        super().__init__()
        self.static_data = None
        
        # Spatial encoding
        self.spatial_net = nn.Sequential(
            nn.Linear(2, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Static feature processor (target + neighbors)
        self.static_net = nn.Sequential(
            nn.Linear(static_dim * (k_neighbors + 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Dynamic feature processor with BatchNorm
        self.dynamic_emb = nn.Sequential(
            nn.Linear(dynamic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Prediction head with learnable scaling
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, prediction_length),
            nn.Tanh()  # Already added as per user
        )
        
        # Learnable scaling parameters
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.shift = nn.Parameter(torch.tensor(0.0))
        
        # Neighbor config
        self.k_neighbors = k_neighbors
        self.static_tree = None

    def build_spatial_index(self, static_data):
        """Store original data and build spatial index with radians"""
        self.static_data = static_data
        stations_order = static_data['station_name'].values
        lats = static_data['latitude'].sel(station_name=stations_order).values
        lons = static_data['longitude'].sel(station_name=stations_order).values
        
        lats_rad = np.radians(lats)
        lons_rad = np.radians(lons)
        
        self.static_tree = BallTree(np.column_stack([lats_rad, lons_rad]), metric='haversine')
        self.station_indices = {name: idx for idx, name in enumerate(stations_order)}

    def forward(self, static, dynamic, lat, lon, all_statics):
        lat_orig = (lat + 1) / 2 * (49 - 24) + 24
        lon_orig = (lon + 1) / 2 * (-66.5 - (-125)) + (-125)
        
        lat_rad = np.radians(lat_orig.detach().cpu().numpy())
        lon_rad = np.radians(lon_orig.detach().cpu().numpy())
        spatial_rad = np.column_stack([lat_rad, lon_rad])
        
        spatial = torch.cat([lat_orig, lon_orig], dim=-1)
        spatial_feat = self.spatial_net(spatial)
        
        batch_size = static.size(0)
        neighbor_statics = []
        
        for i in range(batch_size):
            latlon = spatial_rad[i].reshape(1, -1)
            _, neighbors = self.static_tree.query(latlon, k=self.k_neighbors + 1)
            valid_neighbors = neighbors[0][1:self.k_neighbors + 1]
            
            if len(valid_neighbors) < self.k_neighbors:
                valid_neighbors = np.pad(
                    valid_neighbors,
                    (0, self.k_neighbors - len(valid_neighbors)),
                    mode='edge'
                )
            
            neighbor_data = all_statics[valid_neighbors].flatten()
            neighbor_data = torch.tensor(neighbor_data).float().to(static.device)
            neighbor_statics.append(neighbor_data)
        
        combined_static = torch.cat([static, torch.stack(neighbor_statics)], dim=-1)
        static_feat = self.static_net(combined_static)
        
        # Process dynamic features
        dynamic_feat = self.dynamic_emb(dynamic)
        dynamic_feat += static_feat.unsqueeze(1)
        
        temporal_feat = self.transformer(dynamic_feat)
        temporal_feat = temporal_feat.mean(dim=1)
        
        # Apply scaling
        raw_output = self.output(temporal_feat + static_feat)
        return raw_output * self.scale + self.shift