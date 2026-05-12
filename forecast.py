import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from data_fetcher import fetch_recent_data
from preprocess import preprocess_data
from model import WeatherTransformer, get_device
import matplotlib.pyplot as plt

if __name__ == "__main__":
    LAT = os.getenv("LAT", "51.5074")
    LON = os.getenv("LON", "-0.1278")
    
    print("Fetching latest real-time data for seed (last 7 days)...")
    # Fetch 7 days + current day to ensure we have 168h
    df_latest = fetch_recent_data(LAT, LON, past_days=7)
    
    # Preprocess to get the scaler
    _, _, _, _, scaler = preprocess_data("weather_data.csv", seq_length=168, horizon=24)
    
    cols = ['temp', 'humidity', 'pressure', 'wind_speed', 'wind_sin', 'wind_cos', 'rain', 'cloud_cover', 'solar_rad']
    
    # Process latest data to match features
    df_latest['wind_sin'] = np.sin(2 * np.pi * df_latest['wind_deg'] / 360)
    df_latest['wind_cos'] = np.cos(2 * np.pi * df_latest['wind_deg'] / 360)
    
    # Ensure we have exactly 168 hours of seed data
    seed_data = df_latest[cols].values[-168:]
    if len(seed_data) < 168:
        print(f"Warning: Only {len(seed_data)} hours of seed data available. Padding with first row.")
        padding = np.tile(seed_data[0], (168 - len(seed_data), 1))
        seed_data = np.vstack([padding, seed_data])
        
    scaled_seed = scaler.transform(seed_data)
    seed_tensor = torch.FloatTensor(scaled_seed).unsqueeze(0)
    
    # Load Model (Now with 8 outputs per step)
    model = WeatherTransformer(input_size=len(cols), d_model=128, nhead=8, num_layers=3, output_size=24*8)
    model.load_state_dict(torch.load("weather_transformer.pth"))
    model.to(get_device())
    model.eval()
    
    print("Generating 24-hour direct multi-step forecast with Transformer...")
    with torch.no_grad():
        preds_scaled_flat = model(seed_tensor.to(get_device())).cpu().numpy()
        # Reshape to [24, 8]
        preds_scaled = preds_scaled_flat.reshape(24, 8)
    
    # Inverse scaling for all variables
    # Order of targets: [temp(0), humidity(1), wind_speed(2), wind_sin(3), wind_cos(4), rain(5), cloud_cover(6), solar_rad(7)]
    # Mapping to features (cols): 0->0, 1->1, 2->3, 3->4, 4->5, 5->6, 6->7, 7->8
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
    
    temps = unscaled[:, 0]
    hums = unscaled[:, 1]
    wind_speeds = np.maximum(0, unscaled[:, 3])
    wind_degs = (np.arctan2(unscaled[:, 4], unscaled[:, 5]) * 180 / np.pi) % 360
    rains = np.maximum(0, unscaled[:, 6])
    clouds = np.clip(unscaled[:, 7], 0, 100)
    solars = np.maximum(0, unscaled[:, 8])
    
    forecast_times = [df_latest.index[-1] + timedelta(hours=i+1) for i in range(24)]
    forecast_df = pd.DataFrame({
        'Time': forecast_times,
        'Temp (°C)': temps,
        'Humidity (%)': hums,
        'Wind Spd (km/h)': wind_speeds,
        'Wind Deg (°)': wind_degs,
        'Rain (mm)': rains,
        'Cloud (%)': clouds,
        'Solar (W/m²)': solars
    }).round(2)
    
    print("\nNext 24 Hours Transformer Forecast (Starting from Now):")
    
    # Print recent actual data for context in Cyan
    CYAN = "\033[36m"
    RESET = "\033[0m"
    
    recent_actuals = df_latest[['temp', 'humidity', 'wind_speed', 'wind_deg', 'rain', 'cloud_cover', 'solar_rad']].tail(3).copy().astype(float)
    recent_actuals.columns = ['Temp (°C)', 'Humidity (%)', 'Wind Spd (km/h)', 'Wind Deg (°)', 'Rain (mm)', 'Cloud (%)', 'Solar (W/m²)']
    pd.options.display.float_format = '{:.2f}'.format
    print(f"{CYAN}Recent Actual Weather (Last 3 Hours):")
    print(recent_actuals)
    print(f"{RESET}")
    
    print(forecast_df)
    
    # Plotting
    fig, axes = plt.subplots(7, 1, figsize=(12, 28))
    
    # Temp
    axes[0].plot(df_latest.index[-24:], df_latest['temp'][-24:], label='Past 24h Actual')
    axes[0].plot(forecast_times, temps, label='Forecast', linestyle='--')
    axes[0].set_title("Temperature (°C)")
    axes[0].legend()
    
    # Humidity
    axes[1].plot(df_latest.index[-24:], df_latest['humidity'][-24:], label='Past 24h Actual', color='green')
    axes[1].plot(forecast_times, hums, label='Forecast', linestyle='--', color='green')
    axes[1].set_title("Humidity (%)")
    axes[1].legend()
    
    # Wind Speed
    axes[2].plot(df_latest.index[-24:], df_latest['wind_speed'][-24:], label='Past 24h Actual', color='teal')
    axes[2].plot(forecast_times, wind_speeds, label='Forecast', linestyle='--', color='teal')
    axes[2].set_title("Wind Speed (km/h)")
    axes[2].legend()
    
    # Wind Direction
    axes[3].plot(df_latest.index[-24:], df_latest['wind_deg'][-24:], label='Past 24h Actual', color='purple')
    axes[3].plot(forecast_times, wind_degs, label='Forecast', linestyle='--', color='purple')
    for i in range(0, 24, 3):
        try:
            axes[3].annotate('', xy=(df_latest.index[-24+i], df_latest['wind_deg'][-24+i]),
                         xytext=(0, -10), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', color='purple', lw=1.5, mutation_scale=15),
                         rotation=df_latest['wind_deg'][-24+i])
        except: pass
        axes[3].annotate('', xy=(forecast_times[i], wind_degs[i]),
                     xytext=(0, -10), textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5, mutation_scale=15),
                     rotation=wind_degs[i])
    axes[3].set_title("Wind Direction (°) ")
    axes[3].legend()

    # Rain
    axes[4].bar(df_latest.index[-24:], df_latest['rain'][-24:], label='Past 24h Actual', color='blue', alpha=0.5)
    axes[4].bar(forecast_times, rains, label='Forecast', color='blue', alpha=0.3)
    axes[4].set_title("Rain (mm)")
    axes[4].legend()

    # Cloud Cover
    axes[5].plot(df_latest.index[-24:], df_latest['cloud_cover'][-24:], label='Past 24h Actual', color='gray')
    axes[5].plot(forecast_times, clouds, label='Forecast', linestyle='--', color='gray')
    axes[5].set_title("Cloud Cover (%)")
    axes[5].legend()

    # Solar Radiation
    axes[6].plot(df_latest.index[-24:], df_latest['solar_rad'][-24:], label='Past 24h Actual', color='orange')
    axes[6].plot(forecast_times, solars, label='Forecast', linestyle='--', color='orange')
    axes[6].set_title("Solar Radiation (W/m²)")
    axes[6].legend()
    
    plt.tight_layout()
    plt.savefig("multi_forecast_plot.png")
    print("\nUpdated multi-variate forecast plot saved to multi_forecast_plot.png")
