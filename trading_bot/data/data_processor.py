import pandas as pd
import numpy as np
import os

# Paths for data storage
DATA_PATH = "trading_bot/data/market_data.csv"
X_SAVE_PATH = "trading_bot/data/X.npy"
Y_SAVE_PATH = "trading_bot/data/y.npy"

def compute_rsi(series, period=14):
    """
    Compute the Relative Strength Index (RSI).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_indicators(df):
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['close'])
    df['volatility'] = df['close'].rolling(window=10).std()
    df['momentum'] = df['close'] - df['close'].shift(4)  # 4-period momentum
    df.dropna(inplace=True)
    return df

def process_data():
    """
    Load market data, compute indicators, and save as processed arrays.
    """
    if not os.path.exists(DATA_PATH):
        print(f"❌ Market data file not found: {DATA_PATH}")
        return

    # Load market data
    df = pd.read_csv(DATA_PATH)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Compute indicators
    df = compute_indicators(df)

    # Define input (X) and target (y)
    X = df[["SMA_20", "SMA_50", "RSI", "volatility", "momentum"]].values
    y = np.where(df["close"].shift(-1) > df["close"], 1, 0)  # 1 if price rises, else 0

    # Save processed data
    np.save(X_SAVE_PATH, X)
    np.save(Y_SAVE_PATH, y)

    print(f"✅ Processed data saved: {X_SAVE_PATH}, {Y_SAVE_PATH}")

if __name__ == "__main__":
    process_data()