import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Open-Meteo doesn't require an API key for non-commercial use
LAT = os.getenv("LAT", "51.5074")
LON = os.getenv("LON", "-0.1278")

def fetch_historical_weather(lat, lon, days=30):
    """
    Fetch historical weather data from Open-Meteo (Free, No Key required).
    """
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation,cloud_cover,shortwave_radiation",
        "timezone": "auto"
    }
    
    print(f"Requesting data from Open-Meteo for {lat}, {lon}...")
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        hourly = data['hourly']
        
        df = pd.DataFrame({
            "dt": pd.to_datetime(hourly['time']),
            "temp": hourly['temperature_2m'],
            "humidity": hourly['relative_humidity_2m'],
            "pressure": hourly['surface_pressure'],
            "wind_speed": hourly['wind_speed_10m'],
            "wind_deg": hourly['wind_direction_10m'],
            "rain": hourly['precipitation'],
            "cloud_cover": hourly['cloud_cover'],
            "solar_rad": hourly['shortwave_radiation']
        })
        
        df.set_index('dt', inplace=True)
        return df
    else:
        raise Exception(f"Error fetching data from Open-Meteo: {response.status_code} - {response.text}")

def fetch_recent_data(lat, lon, past_days=2):
    """
    Fetch recent weather data including the current hour
    using the Open-Meteo Forecast API.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": past_days,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation,cloud_cover,shortwave_radiation",
        "timezone": "auto"
    }
    
    print(f"Requesting recent data from Open-Meteo for {lat}, {lon}...")
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        hourly = data['hourly']
        
        df = pd.DataFrame({
            "dt": pd.to_datetime(hourly['time']),
            "temp": hourly['temperature_2m'],
            "humidity": hourly['relative_humidity_2m'],
            "pressure": hourly['surface_pressure'],
            "wind_speed": hourly['wind_speed_10m'],
            "wind_deg": hourly['wind_direction_10m'],
            "rain": hourly['precipitation'],
            "cloud_cover": hourly['cloud_cover'],
            "solar_rad": hourly['shortwave_radiation']
        })
        
        # Filter to only include up to the current hour
        current_time = datetime.now()
        df = df[df['dt'] <= current_time]
        
        df.set_index('dt', inplace=True)
        return df
    else:
        raise Exception(f"Error fetching recent data: {response.status_code} - {response.text}")

if __name__ == "__main__":
    try:
        lat, lon = float(LAT), float(LON)
        # Fetching 3650 days (10 years) for a comprehensive training set
        weather_df = fetch_historical_weather(lat, lon, days=3650)
        print(weather_df.head())
        weather_df.to_csv("weather_data.csv")
        print(f"Successfully saved {len(weather_df)} records to weather_data.csv")
    except Exception as e:
        print(f"An error occurred: {e}")
