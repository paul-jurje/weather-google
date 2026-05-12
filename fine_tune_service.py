import subprocess
import threading
import os
from data_fetcher import fetch_recent_data

# Global state to track the background fine-tuning task
ft_status = {
    "status": "idle",
    "message": "Ready to fine-tune.",
    "is_running": False
}

def _run_fine_tune_thread(epochs, days):
    global ft_status
    ft_status["status"] = "running"
    ft_status["message"] = f"Fine-tuning started ({epochs} epochs, {days} days of data)..."
    ft_status["is_running"] = True

    try:
        # Execute fine_tune.py in a separate process to avoid blocking the server loop
        # and to cleanly manage PyTorch GPU/MPS memory allocation.
        result = subprocess.run(
            ["python3", "fine_tune.py", "--epochs", str(epochs), "--days", str(days)],
            capture_output=True,
            text=True,
            check=True
        )
        ft_status["status"] = "success"
        ft_status["message"] = "Fine-tuning completed successfully! New weights saved."
    except subprocess.CalledProcessError as e:
        ft_status["status"] = "error"
        ft_status["message"] = f"Training failed. Error: {e.stderr or e.stdout}"
    except Exception as e:
        ft_status["status"] = "error"
        ft_status["message"] = f"Unexpected error: {str(e)}"
    finally:
        ft_status["is_running"] = False

def start_fine_tune(epochs=5, days=14):
    global ft_status
    if ft_status["is_running"]:
        return False, "Fine-tuning is already in progress."

    # Start the fine-tuning process in a background thread
    thread = threading.Thread(target=_run_fine_tune_thread, args=(epochs, days))
    thread.daemon = True
    thread.start()

    return True, "Fine-tuning job started in the background."

def get_ft_status():
    return ft_status

def refresh_data():
    try:
        LAT = os.getenv("LAT", "51.5074")
        LON = os.getenv("LON", "-0.1278")
        # Call the data fetcher to pull the latest Open-Meteo data
        fetch_recent_data(LAT, LON, past_days=7)
        return True, "Weather data cache refreshed successfully."
    except Exception as e:
        return False, f"Failed to refresh data: {str(e)}"