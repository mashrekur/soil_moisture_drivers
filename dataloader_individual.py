import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader

class IndividualStationDataset(Dataset):
    def __init__(self, station_id, static_path, dynamic_path, target_path,
                 seq_len=14, pred_len=2, time_range=None):
        
        # Load static data (with context manager)
        with xr.open_dataset(static_path) as ds:
            static = ds.sel(station_name=station_id).load()
            self.static = static.drop_vars(['latitude', 'longitude'])
            self.static_values = self.static.to_array().values  # Store in memory
        
        # Load dynamic data (with context manager)
        with xr.open_dataset(dynamic_path) as ds:
            self.dynamic = ds.sel(station=station_id).load()
            if time_range:
                self.dynamic = self.dynamic.sel(time=slice(*time_range))
        
        # Load target data (with context manager)
        with xr.open_dataset(target_path) as ds:
            self.target = ds.sel(station=station_id).load()
            if time_range:
                self.target = self.target.sel(time=slice(*time_range))
        
        # Store coordinates
        self.lat = float(static.latitude)
        self.lon = float(static.longitude)
        self.times = self.target.time.values
        self.valid_starts = np.arange(len(self.times) - (seq_len + pred_len) + 1)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.seq_len + self.pred_len
        
        # Dynamic features
        dynamic = self.dynamic.isel(time=slice(start, end))
        dynamic = torch.FloatTensor(dynamic.to_array().values.T)
        
        # Targets
        target = self.target.isel(time=slice(start+self.seq_len, end))
        target = torch.FloatTensor(target.soil_moisture.values)
        
        # Static features
        static = torch.FloatTensor(self.static_values)
        
        # Normalized coordinates
        lat_norm = (self.lat - 24) / (49 - 24) * 2 - 1
        lon_norm = (self.lon - (-125)) / (-66.5 - (-125)) * 2 - 1
        
        # Timestamps
        times = self.times[start:end]
        times_unix = (times - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        times_unix = times_unix.astype(np.float32)
        
        return {
            'static': static,
            'dynamic': dynamic,
            'target': target,
            'lat': torch.FloatTensor([lat_norm]),
            'lon': torch.FloatTensor([lon_norm]),
            'times': torch.FloatTensor(times_unix)
        }

def get_station_loaders(station_id, config):
    train_set = IndividualStationDataset(
        station_id=station_id,
        static_path=config['static_path'],
        dynamic_path=config['dynamic_path'],
        target_path=config['target_path'],
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        time_range=config['train_range']
    )
    
    test_set = IndividualStationDataset(
        station_id=station_id,
        static_path=config['static_path'],
        dynamic_path=config['dynamic_path'],
        target_path=config['target_path'], 
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        time_range=config['test_range']
    )
    
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False)
    
    return train_loader, test_loader