import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from data_fetcher import fetch_historical_weather
from preprocess import preprocess_data
from model import WeatherTransformer, get_device
import argparse
import time

def fine_tune_model(X_ft, y_ft, params):
    device = get_device()
    print(f"Fine-tuning on device: {device}")
    
    horizon = y_ft.shape[1]
    num_targets = y_ft.shape[2]
    output_size = horizon * num_targets
    
    # Initialize and load current weights
    model = WeatherTransformer(params['input_size'], params['d_model'], 
                               params['nhead'], params['num_layers'], output_size).to(device)
    
    if os.path.exists("weather_transformer.pth"):
        print("Loading existing weights for fine-tuning...")
        model.load_state_dict(torch.load("weather_transformer.pth"))
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    
    y_ft_flat = y_ft.view(y_ft.shape[0], -1)
    
    dataset = TensorDataset(X_ft, y_ft_flat)
    loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
    
    start_ft_time = time.time()
    print(f"\nStarting fine-tuning at: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 50)

    model.train()
    for epoch in range(params['epochs']):
        epoch_start_time = time.time()
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        epoch_losses = []
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            
        epoch_duration = time.time() - epoch_start_time
        print(f"[{timestamp}] Fine-Tuning Epoch [{epoch+1}/{params['epochs']}] - Loss: {sum(epoch_losses)/len(epoch_losses):.6f} - Duration: {epoch_duration:.2f}s")
        
    total_time = time.time() - start_ft_time
    print("-" * 50)
    print(f"Fine-Tuning Complete at: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total Time: {total_time:.2f}s")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune the Weather Transformer model.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of fine-tuning epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for fine-tuning (default: 1e-5)')
    parser.add_argument('--days', type=int, default=14, help='Number of recent days to fetch for fine-tuning (default: 14)')
    args = parser.parse_args()

    LAT = os.getenv("LAT", "51.5074")
    LON = os.getenv("LON", "-0.1278")
    
    print(f"Fetching last {args.days} days of actual data for fine-tuning...")
    # Fetch actual historical data
    df_recent = fetch_historical_weather(LAT, LON, days=args.days)
    
    # Preprocess (reuse the scaler from the main 10-year training)
    _, _, _, _, scaler = preprocess_data("weather_data.csv", seq_length=168, horizon=24)
    
    cols = ['temp', 'humidity', 'pressure', 'wind_speed', 'wind_sin', 'wind_cos', 'rain', 'cloud_cover', 'solar_rad']
    
    # Manual preprocessing of recent data
    df_recent['wind_sin'] = np.sin(2 * np.pi * df_recent['wind_deg'] / 360)
    df_recent['wind_cos'] = np.cos(2 * np.pi * df_recent['wind_deg'] / 360)
    df_recent['rain'] = df_recent['rain'].fillna(0)
    
    data_values = df_recent[cols].values
    scaled_data = scaler.transform(data_values)
    
    from preprocess import create_sequences
    target_indices = [0, 1, 3, 4, 5, 6, 7, 8]
    X_ft, y_ft = create_sequences(scaled_data, seq_length=168, horizon=24, target_indices=target_indices)
    
    if len(X_ft) == 0:
        print(f"Not enough data in the last {args.days} days to create sequences.")
    else:
        params = {
            'input_size': len(cols),
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'lr': args.lr,
            'batch_size': 16,
            'epochs': args.epochs
        }
        
        updated_model = fine_tune_model(torch.FloatTensor(X_ft), torch.FloatTensor(y_ft), params)
        torch.save(updated_model.state_dict(), "weather_transformer.pth")
        print("Fine-tuning complete. Model updated with recent trends.")
