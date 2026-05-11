import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(file_path, hours=500):
    start_time = datetime.now() - timedelta(hours=hours)
    data = []
    
    for i in range(hours):
        dt = start_time + timedelta(hours=i)
        # Simulate temperature with a daily cycle and some noise
        hour = dt.hour
        temp = 15 + 10 * np.sin(2 * np.pi * (hour - 6) / 24) + np.random.normal(0, 1)
        humidity = 50 + 20 * np.cos(2 * np.pi * (hour - 6) / 24) + np.random.normal(0, 5)
        pressure = 1013 + np.random.normal(0, 2)
        wind_speed = 5 + np.random.normal(0, 2)
        wind_deg = (hour * 15 + np.random.normal(0, 10)) % 360
        rain = max(0, np.random.normal(0.1, 0.5)) if np.random.random() > 0.8 else 0
        
        data.append({
            'dt': dt.strftime('%Y-%m-%d %H:%M:%S'),
            'temp': temp,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'wind_deg': wind_deg,
            'rain': rain
        })
        
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Synthetic data saved to {file_path}")

if __name__ == "__main__":
    generate_synthetic_data("weather_data.csv")
