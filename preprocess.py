import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

def create_sequences(data, seq_length, horizon=24, target_indices=[0, 1, 3, 4, 5, 6, 7, 8]):
    """
    target_indices: Indices of columns we want to predict.
    """
    xs = []
    ys = []
    for i in range(len(data) - seq_length - horizon + 1):
        x = data[i:(i + seq_length)]
        y = data[(i + seq_length):(i + seq_length + horizon), target_indices] 
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def get_feature_data(df):
    """Consistent feature engineering for a DataFrame."""
    df = df.copy()
    if 'wind_deg' in df.columns:
        df['wind_sin'] = np.sin(2 * np.pi * df['wind_deg'] / 360)
        df['wind_cos'] = np.cos(2 * np.pi * df['wind_deg'] / 360)
    
    if 'rain' in df.columns:
        df['rain'] = df['rain'].fillna(0)
    
    # Order: temp(0), humidity(1), pressure(2), wind_speed(3), wind_sin(4), wind_cos(5), rain(6), cloud_cover(7), solar_rad(8)
    cols = ['temp', 'humidity', 'pressure', 'wind_speed']
    if 'wind_sin' in df.columns:
        cols.extend(['wind_sin', 'wind_cos'])
    if 'rain' in df.columns:
        cols.append('rain')
    if 'cloud_cover' in df.columns:
        cols.append('cloud_cover')
    if 'solar_rad' in df.columns:
        cols.append('solar_rad')
    
    return df[cols], cols

def get_scaler(file_path):
    """Lightweight function to get a fitted scaler without sequence generation."""
    df = pd.read_csv(file_path, index_col='dt')
    data_df, _ = get_feature_data(df)
    scaler = MinMaxScaler()
    scaler.fit(data_df.values)
    return scaler

def preprocess_data(file_path, seq_length=168, horizon=24):
    df = pd.read_csv(file_path, index_col='dt')
    data_df, cols = get_feature_data(df)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_df.values)

    # Predict: temp, humidity, wind_speed, wind_sin, wind_cos, rain, cloud_cover, solar_rad
    target_vars = ['temp', 'humidity', 'wind_speed', 'wind_sin', 'wind_cos', 'rain', 'cloud_cover', 'solar_rad']
    target_indices = [i for i, col in enumerate(cols) if col in target_vars]
    
    X, y = create_sequences(scaled_data, seq_length, horizon=horizon, target_indices=target_indices)

    # Split into train/test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return (torch.FloatTensor(X_train), torch.FloatTensor(y_train), 
            torch.FloatTensor(X_test), torch.FloatTensor(y_test), 
            scaler)

if __name__ == "__main__":
    try:
        X_train, y_train, X_test, y_test, scaler = preprocess_data("weather_data.csv")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
    except FileNotFoundError:
        print("weather_data.csv not found. Run data_fetcher.py first.")
