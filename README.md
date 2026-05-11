# Weather Prediction AI Model (Advanced Transformer Edition)

A professional-grade, local weather forecasting system using a **Transformer Encoder** architecture, optimized for Apple Silicon (M1).

## Advanced Features

- **Transformer Architecture:** Replaces LSTMs with self-attention mechanisms to better capture complex temporal dependencies.
- **Deep 7-Day Context:** The model analyzes the last **168 hours (1 week)** of weather history to predict future trends.
- **8-Variable Multi-Variate Prediction:** Simultaneously predicts Temperature, Humidity, Wind Speed, Wind Direction (Periodic sin/cos), Rain, **Cloud Cover (%)**, and **Solar Radiation (W/m²)**.
- **Direct Multi-Step Forecasting:** Predicts the entire 24-hour horizon in a single pass, significantly reducing the "drift" associated with recursive models.
- **Massive 10-Year Dataset:** Trained on 87,000+ hourly records for superior stability and seasonal awareness.
- **M1 GPU Accelerated:** Full Metal Performance Shaders (MPS) support for high-speed training and inference.

## Project Structure

- `data_fetcher.py`: Fetches 10 years of historical data and real-time seed data from Open-Meteo.
- `preprocess.py`: Processes 168h windows and 24h targets with cyclical encoding.
- `model.py`: Implements the `WeatherTransformer` with Positional Encoding.
- `train.py`: Optimized training loop for multi-step Transformer learning with real-time timestamps.
- `forecast.py`: Direct inference script that generates a 24-hour massive multi-variate forecast.
- `fine_tune.py`: Incremental learning script for daily model adjustments.

## Getting Started

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Location (Optional):**
   ```env
   LAT=51.5074
   LON=-0.1278
   ```
3. **Train the Model:**
   First, fetch the 10-year dataset, then run the training. You can specify the number of epochs via the CLI:
   ```bash
   # Quick smoke test (1 epoch)
   python train.py --epochs 1

   # Full training blast (100 epochs)
   python train.py --epochs 100
   ```
   *Note: Full training on 10 years of data with a Transformer takes ~5-6 hours on an M1 Pro (@ ~3.5 min/epoch).*

## How to Get Predictions

Run the following command to get an instant 24-hour forecast starting from the current hour:
```bash
python forecast.py
```

### Outputs
- **`multi_forecast_plot.png`**: A 7-panel visualization including Temperature, Humidity, Wind Speed, Wind Direction (with rotated arrows), Rain, Cloud Cover, and Solar Radiation.
- **Terminal Table**: Precise hourly values for all 8 predicted variables (preceded by the last 3 hours of actual data in Cyan for context).

## Continuous Learning (Daily Maintenance)

To keep the model adjusted to the latest local weather trends and anomalies, you can run the fine-tuning script daily:

```bash
# Standard maintenance (5 epochs, 14 days of recent data)
python fine_tune.py

# Aggressive maintenance (10 epochs, 30 days of recent data)
python fine_tune.py --epochs 10 --days 30
```

**What this does:**
1. Fetches the actual weather for the recent period (default 14 days).
2. Loads your existing Transformer weights.
3. Performs a brief, low-learning-rate training pass to "nudge" the model towards recent patterns.
4. Overwrites `weather_transformer.pth` with the updated, more adaptive version.

## Hardware Optimization

The system utilizes the M1 GPU via the `mps` backend. Transformers are highly parallelizable, making them exceptionally fast on Apple Silicon.
