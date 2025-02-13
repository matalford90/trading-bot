from fastapi import FastAPI
from fastapi.responses import FileResponse
import os
import pandas as pd
from trading_bot.utils.logger import logger

app = FastAPI(title="Trading Bot API", version="1.0")

# File path for trading history (adjust as needed)
TRADING_HISTORY_FILE = "trading_bot/data/market_data.csv"

@app.get("/status")
def get_status():
    """
    Get the current status of the trading bot.
    This data should be updated dynamically.
    """
    status = {
        "active_trades": 5,
        "last_update": "2025-02-11T12:34:56Z",
        "model_loss": 0.025,
        "sentiment": "Bullish"
    }
    logger.info("Status endpoint accessed.")
    return status

@app.get("/trading_history")
def get_trading_history():
    """
    Retrieve the trading history.
    Returns the first few rows of the CSV file.
    """
    if not os.path.exists(TRADING_HISTORY_FILE):
        logger.warning("Trading history file not found.")
        return {"error": "Trading history file not found."}

    try:
        df = pd.read_csv(TRADING_HISTORY_FILE)
        logger.info("Trading history retrieved successfully.")
        return df.head(10).to_dict(orient="records")  # Returns first 10 records
    except Exception as e:
        logger.error(f"Error reading trading history: {e}")
        return {"error": "Failed to retrieve trading history."}

@app.get("/download")
def download_csv():
    """
    Provide a CSV download of the trading history.
    """
    if os.path.exists(TRADING_HISTORY_FILE):
        logger.info("Trading history file downloaded.")
        return FileResponse(TRADING_HISTORY_FILE, media_type='text/csv', filename="trading_history.csv")
    else:
        logger.warning("Download attempted, but file not found.")
        return {"error": "File not found"}