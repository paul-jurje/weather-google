import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

export interface WeatherData {
  time: string;
  temp_c: number;
  humidity_pct: number;
  wind_deg: number;
  wind_speed: number;
  rain_mm: number;
  cloud_pct: number;
  solar_wm2: number;
}

export interface ForecastResponse {
  forecast: WeatherData[];
  recent: WeatherData[];
}

export const fetchForecast = async (): Promise<ForecastResponse> => {
  const response = await axios.get<any>(`${API_BASE_URL}/forecast`);

  // Map the backend Pandas DataFrame column names to the frontend interface
  const mappedForecast = (response.data.forecast || []).map((item: any) => ({
    time: item['Time'],
    temp_c: item['Temp (°C)'],
    humidity_pct: item['Humidity (%)'],
    wind_speed: item['Wind Spd (km/h)'],
    wind_deg: item['Wind Deg (°)'],
    rain_mm: item['Rain (mm)'],
    cloud_pct: item['Cloud (%)'],
    solar_wm2: item['Solar (W/m²)'],
  }));

  // Fallback: Since the backend doesn't return 'recent' data currently,
  // we use the first forecast hour as our current conditions to prevent crashes.
  const currentFallback = mappedForecast.length > 0 ? mappedForecast[0] : {} as any;

  return {
    forecast: mappedForecast,
    recent: [currentFallback],
  };
};

export const refreshForecastData = async (): Promise<any> => {
  const response = await axios.post(`${API_BASE_URL}/data/refresh`);
  return response.data;
};

export const startFineTuning = async (): Promise<any> => {
  const response = await axios.post(`${API_BASE_URL}/fine-tune`);
  return response.data;
};

export interface FineTuneStatus {
  status: 'idle' | 'running' | 'success' | 'error';
  message: string;
  is_running: boolean;
}

export const getFineTuneStatus = async (): Promise<FineTuneStatus> => {
  const response = await axios.get<FineTuneStatus>(`${API_BASE_URL}/fine-tune/status`);
  return response.data;
};
