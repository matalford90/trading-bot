import sys
import os
import time
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from trading_bot.models.ml_model import TradingModel  # Adjusted import path

# Check for Apple MPS Support (for M1/M2 MacBooks)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Load KuCoin API Credentials safely
try:
    from trading_bot.config import KUCOIN_API_KEY, KUCOIN_SECRET, KUCOIN_PASSPHRASE
    if not KUCOIN_API_KEY or not KUCOIN_SECRET or not KUCOIN_PASSPHRASE:
        raise ValueError("⚠️ KuCoin API credentials are missing!")
    print("✅ KuCoin API credentials loaded successfully!")
except (ModuleNotFoundError, ValueError) as e:
    print(f"⚠️ Error loading API credentials: {e}")
    KUCOIN_API_KEY, KUCOIN_SECRET, KUCOIN_PASSPHRASE = None, None, None


class KucoinDataFetcher:
    """ Fetch real-time candlestick data from KuCoin Futures API """

    def __init__(self, symbol="XBTUSDTM"):
        self.base_url = "https://api-futures.kucoin.com"
        self.headers = {
            "KC-API-KEY": KUCOIN_API_KEY,
            "KC-API-SECRET": KUCOIN_SECRET,
            "KC-API-PASSPHRASE": KUCOIN_PASSPHRASE,
            "KC-API-KEY-VERSION": "2"
        }
        self.symbol = symbol.upper()

    def get_candlesticks(self, interval="5", limit=1000):
        """ Fetches recent candlestick (OHLCV) data for KuCoin Futures. """
        endpoint = "/api/v1/kline/query"
        current_time = int(time.time() * 1000)  
        start_time = current_time - (limit * int(interval) * 60 * 1000)  

        params = {
            "symbol": self.symbol,
            "granularity": str(interval),
            "from": start_time,
            "to": current_time
        }

        print(f"🔄 Fetching {limit} candlesticks for {self.symbol} at {interval}m intervals...")

        try:
            response = requests.get(self.base_url + endpoint, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            if "code" in data and data["code"] != "200000":
                print(f"⚠️ KuCoin API Error: {data.get('msg', 'Unknown Error')}")
                return None, None

            if "data" not in data or not data["data"]:
                print("⚠️ Error fetching data: No data received.")
                return None, None

            print(f"✅ Received {len(data['data'])} data points.")
            return self.format_data(data["data"])
        except requests.RequestException as e:
            print(f"❌ Error fetching KuCoin data: {e}")
            return None, None

    def format_data(self, raw_data):
        """ Standardizes raw candlestick data (Z-score Normalization) """
        features, labels = [], []
        prices = np.array([float(candle[4]) for candle in raw_data])  # Closing prices

        mean, std = prices.mean(), prices.std()  # Compute mean & std for standardization

        for candle in raw_data:
            if len(candle) < 6:
                print(f"⚠️ Skipping malformed data entry: {candle}")
                continue  

            timestamp, open_price, high, low, close, volume = map(float, candle[:6])

            standardized_features = [
                (open_price - mean) / std,
                (high - mean) / std,
                (low - mean) / std,
                (close - mean) / std,
                volume / max(1, volume)
            ]

            labels.append([(close - mean) / std])  # Standardized labels
            features.append(standardized_features)

        return features, labels


def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.0001):
    """ Trains the model with advanced optimizations """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    save_dir = "trading_bot/models"
    os.makedirs(save_dir, exist_ok=True)  # ✅ Fix: Ensure model save directory exists
    save_path = os.path.join(save_dir, "trading_model_best.pth")

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_features, val_labels in val_loader:
                val_features, val_labels = val_features.to(device), val_labels.to(device)
                val_predictions = model(val_features)
                val_loss += criterion(val_predictions, val_labels).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"🔄 Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"💾 Model checkpoint saved: {save_path}")

    print("🎉 Training Completed Successfully!")


# 🚀 **Main Execution - Continuous Model Training**
if __name__ == "__main__":
    print("\n🚀 Starting Continuous Trading Model Training...\n")

    MODEL_PATH = "trading_bot/models/trading_model_latest.pth"
    TRAIN_INTERVAL = 3600  # Train every hour

    while True:
        print("\n🔄 Fetching Fresh Data & Updating Model...\n")

        # Fetch New Market Data
        fetcher = KucoinDataFetcher("XBTUSDTM")
        features, labels = fetcher.get_candlesticks(interval="5", limit=1000)

        if not features or not labels:
            print("⚠️ No new data received, skipping this cycle.")
        else:
            print(f"✅ New data received. Features: {len(features)}, Labels: {len(labels)}")

            # Load Existing Model (or create a new one)
            model = TradingModel(input_size=5, hidden_size=64, output_size=1).to(device)
            if os.path.exists(MODEL_PATH):
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                print(f"✅ Loaded existing model: {MODEL_PATH}")

            # Convert New Data to Tensor & Wrap in DataLoader
            features_tensor = torch.tensor(features, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.float32)

            dataset = TensorDataset(features_tensor, labels_tensor)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64)

            # Train Model
            train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.0001)

            # Save Updated Model
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"💾 Model Updated: {MODEL_PATH}")

        print(f"⏳ Waiting {TRAIN_INTERVAL // 60} minutes before next training cycle...\n")
        time.sleep(TRAIN_INTERVAL)