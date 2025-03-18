import os
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import xarray as xr
from ViT_LSTM_individual import EnhancedSpatiotemporalModel
from dataloader_individual import get_station_loaders
from sklearn.metrics import r2_score, mean_squared_error

# New quantile loss function
def quantile_loss(preds, target, quantiles=[0.1, 0.5, 0.9]):
    losses = []
    for q in quantiles:
        errors = target - preds
        losses.append(torch.max((q-1)*errors, q*errors).unsqueeze(1))
    return torch.mean(torch.cat(losses, dim=1))

def train_station(station_id, config, device='cuda'):
    # Create output directory (unchanged)
    output_dir = f"unique_stations/{station_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model (unchanged)
    model = EnhancedSpatiotemporalModel(
        static_dim=28,
        dynamic_dim=3,
        num_heads=8,
        num_layers=4,
        hidden_dim=256,
        ff_dim=512,
        dropout=0.1,
        prediction_length=config['pred_len'],
        k_neighbors=5
    ).to(device)
    
    # Load spatial data (unchanged)
    with xr.open_dataset(config['static_path']) as static_data:
        model.build_spatial_index(static_data)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = quantile_loss  # Only changed line in initialization
    
    # Data loaders (unchanged)
    train_loader, test_loader = get_station_loaders(station_id, config)
    
    # Training variables (unchanged)
    best_r2 = -np.inf
    metrics_history = []
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Station {station_id} Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            # Move data to device (unchanged)
            static = batch['static'].to(device)
            dynamic = batch['dynamic'].to(device)
            target = batch['target'].to(device)
            lat = batch['lat'].to(device)
            lon = batch['lon'].to(device)
            
            # Get static data (unchanged)
            with xr.open_dataset(config['static_path']) as static_data:
                static_np = static_data.drop_vars(['latitude', 'longitude']).to_array().values.T
            
            # Forward pass (unchanged)
            preds = model(static, dynamic, lat, lon, static_np)
            loss = criterion(preds, target)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation (unchanged)
        test_metrics = evaluate(model, test_loader, config['static_path'], device)
        metrics_history.append(test_metrics)
        
        # Save best model (unchanged)
        if test_metrics['r2'] > best_r2:
            best_r2 = test_metrics['r2']
            torch.save(model.state_dict(), f"{output_dir}/best_model.pth")
            save_predictions(model, test_loader, config['static_path'], output_dir, device)
        
        print(f"Test RMSE: {test_metrics['rmse']:.4f} | R²: {test_metrics['r2']:.4f} (Best: {best_r2:.4f})")
    
    # Save metrics and plots (unchanged)
    save_metrics(metrics_history, output_dir)
    plot_predictions(output_dir, station_id, best_r2)
    return test_metrics

# All subsequent functions remain EXACTLY AS ORIGINAL
def evaluate(model, loader, static_path, device):
    model.eval()
    preds, truths = [], []
    
    with torch.no_grad():
        with xr.open_dataset(static_path) as static_data:
            static_np = static_data.drop_vars(['latitude', 'longitude']).to_array().values.T
            
            for batch in loader:
                static = batch['static'].to(device)
                dynamic = batch['dynamic'].to(device)
                lat = batch['lat'].to(device)
                lon = batch['lon'].to(device)
                
                pred = model(static, dynamic, lat, lon, static_np)
                preds.append(pred.cpu().numpy().reshape(-1))
                truths.append(batch['target'].numpy().reshape(-1))
    
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    
    return {
        'rmse': np.sqrt(mean_squared_error(truths, preds)),
        'r2': r2_score(truths, preds)
    }

def save_predictions(model, loader, static_path, output_dir, device):
    model.eval()
    all_preds, all_truths, all_times = [], [], []
    
    with torch.no_grad():
        with xr.open_dataset(static_path) as static_data:
            static_np = static_data.drop_vars(['latitude', 'longitude']).to_array().values.T
            
            for batch in loader:
                static = batch['static'].to(device)
                dynamic = batch['dynamic'].to(device)
                lat = batch['lat'].to(device)
                lon = batch['lon'].to(device)
                
                preds = model(static, dynamic, lat, lon, static_np)
                all_preds.append(preds.cpu().numpy().reshape(-1))
                all_truths.append(batch['target'].numpy().reshape(-1))
                
                times_unix = batch['times'].numpy()
                prediction_times_unix = times_unix[:, -config['pred_len']:].reshape(-1)
                prediction_times = np.datetime64('1970-01-01T00:00:00') + prediction_times_unix * np.timedelta64(1, 's')
                all_times.append(prediction_times)
    
    df = pd.DataFrame({
        'timestamp': np.concatenate(all_times),
        'actual': np.concatenate(all_truths),
        'predicted': np.concatenate(all_preds)
    })
    df.to_csv(f"{output_dir}/predictions.csv", index=False)

def plot_predictions(output_dir, station_id, best_r2):
    df = pd.read_csv(f"{output_dir}/predictions.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    plt.figure(figsize=(15, 5))
    plt.plot(df['timestamp'], df['actual'], label='Actual')
    plt.plot(df['timestamp'], df['predicted'], label='Predicted', alpha=0.7)
    plt.title(f"Station {station_id} - Actual vs Predicted (Best R²: {best_r2:.2f})")
    plt.xlabel("Time")
    plt.ylabel("Soil Moisture")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(f"{output_dir}/predictions_plot.png")
    plt.close()

def save_metrics(metrics_history, output_dir):
    df = pd.DataFrame(metrics_history)
    df['epoch'] = range(1, len(df)+1)
    df.to_csv(f"{output_dir}/training_metrics.csv", index=False)

if __name__ == "__main__":
    config = {
        'static_path': 'Data/analysis_output/12h_means/static_data.nc',
        'dynamic_path': 'Data/analysis_output/12h_means/dynamic_data_12h.nc',
        'target_path': 'Data/analysis_output/12h_means/target_data_12h.nc',
        'seq_len': 14,
        'pred_len': 2,
        'batch_size': 32,
        'epochs': 20,
        'train_range': ('2015-01-01', '2018-12-31'),
        'test_range': ('2019-01-01', '2019-12-31')
    }
    
    os.makedirs("unique_stations", exist_ok=True)
    
    with xr.open_dataset(config['static_path']) as ds:
        all_stations = ds['station_name'].values.tolist()
    
    results = {}
    for station in all_stations:
        print(f"\n{'='*50}")
        print(f"Training {station}")
        try:
            metrics = train_station(station, config)
            results[station] = metrics
        except Exception as e:
            print(f"Failed training {station}: {str(e)}")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    pd.DataFrame.from_dict(results, orient='index').to_csv("unique_stations/all_stations_metrics.csv")