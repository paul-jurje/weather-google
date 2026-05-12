import torch
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from data_fetcher import fetch_recent_data
from preprocess import preprocess_data
from model import WeatherTransformer, get_device

def get_forecast():
    # Load Environment Variables
    LAT = os.getenv("LAT", "51.5074")
    LON = os.getenv("LON", "-0.1278")

    # 1. Fetch real-time data and preprocess (Get scaler)
    df_latest = fetch_recent_data(LAT, LON, past_days=7)
    _, _, _, _, scaler = preprocess_data("weather_data.csv", seq_length=168, horizon=24)

    cols = ['temp', 'humidity', 'pressure', 'wind_speed', 'wind_sin', 'wind_cos', 'rain', 'cloud_cover', 'solar_rad']
    df_latest['wind_sin'] = np.sin(2 * np.pi * df_latest['wind_deg'] / 360)
    df_latest['wind_cos'] = np.cos(2 * np.pi * df_latest['wind_deg'] / 360)

    # Pad if seed data is insufficient
    seed_data = df_latest[cols].values[-168:]
    if len(seed_data) < 168:
        padding = np.tile(seed_data[0], (168 - len(seed_data), 1))
        seed_data = np.vstack([padding, seed_data])

    scaled_seed = scaler.transform(seed_data)
    seed_tensor = torch.FloatTensor(scaled_seed).unsqueeze(0)

    # 2. Run Direct Multi-Step Transformer Inference
    model = WeatherTransformer(input_size=len(cols), d_model=128, nhead=8, num_layers=3, output_size=24*8)
    model.load_state_dict(torch.load("weather_transformer.pth", map_location=get_device()))
    model.to(get_device())
    model.eval()

    with torch.no_grad():
        preds_scaled_flat = model(seed_tensor.to(get_device())).cpu().numpy()
        preds_scaled = preds_scaled_flat.reshape(24, 8)

    # 3. Post-Process (Inverse Scaling)
    dummy = np.zeros((24, len(cols)))
    dummy[:, 0] = preds_scaled[:, 0] # Temp
    dummy[:, 1] = preds_scaled[:, 1] # Humidity
    dummy[:, 3] = preds_scaled[:, 2] # Wind Speed
    dummy[:, 4] = preds_scaled[:, 3] # Wind Sin
    dummy[:, 5] = preds_scaled[:, 4] # Wind Cos
    dummy[:, 6] = preds_scaled[:, 5] # Rain
    dummy[:, 7] = preds_scaled[:, 6] # Cloud Cover
    dummy[:, 8] = preds_scaled[:, 7] # Solar Rad

    unscaled = scaler.inverse_transform(dummy)
    forecast_times = [df_latest.index[-1] + timedelta(hours=i+1) for i in range(24)]

    forecast_df = pd.DataFrame({
        'Time': [t.isoformat() for t in forecast_times], # String serialization for JSON
        'Temp (°C)': unscaled[:, 0],
        'Humidity (%)': unscaled[:, 1],
        'Wind Spd (km/h)': np.maximum(0, unscaled[:, 3]),
        'Wind Deg (°)': (np.arctan2(unscaled[:, 4], unscaled[:, 5]) * 180 / np.pi) % 360,
        'Rain (mm)': np.maximum(0, unscaled[:, 6]),
        'Cloud (%)': np.clip(unscaled[:, 7], 0, 100),
        'Solar (W/m²)': np.maximum(0, unscaled[:, 8])
    }).round(2)

    # 4. Structure into native Python dict, handling any rogue NaNs
    return forecast_df.fillna(0).to_dict(orient="records")