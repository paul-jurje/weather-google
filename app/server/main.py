import sys
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "api"))

# Change the current working directory to the project root so that
# relative paths (like "weather_transformer.pth" or "weather_data.csv") resolve correctly.
os.chdir(parent_dir)

try:
    from forecast_service import get_forecast
    from fine_tune_service import start_fine_tune, get_ft_status, refresh_data
except Exception as e:
    print(f"Warning: Failed to import services. Reason: {e}")
    # Fallback if imports fail
    def get_forecast():
        raise HTTPException(status_code=503, detail="Could not load forecast service")
    def start_fine_tune():
        raise HTTPException(status_code=503, detail="Fine-tune service unavailable")
    def get_ft_status():
        raise HTTPException(status_code=503, detail="Service unavailable")
    def refresh_data():
        raise HTTPException(status_code=503, detail="Refresh service unavailable")

app = FastAPI(title="Weather App Bridge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/forecast")
def api_get_forecast():
    try:
        data = get_forecast()
        return {"status": "success", "forecast": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/fine-tune")
def api_start_fine_tune():
    try:
        success, message = start_fine_tune()
        return {"status": "success" if success else "error", "message": message}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/fine-tune/status")
def api_get_ft_status():
    try:
        return get_ft_status()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/refresh")
def api_refresh_data():
    try:
        success, message = refresh_data()
        return {"status": "success" if success else "error", "message": message}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))