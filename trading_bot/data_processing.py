import pandas as pd

def compute_indicators(candles):
    """
    Compute technical indicators for the given candlestick data.

    Parameters:
      candles : List of candlestick data, each in the format [timestamp, open, high, low, close, volume]

    Returns:
      DataFrame with computed indicators.
    """
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Example indicators
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['close'])
    
    return df

def compute_rsi(series, period=14):
    """
    Compute the Relative Strength Index (RSI) for a given series.

    Parameters:
      series : Pandas Series of closing prices
      period : Period for RSI calculation (default is 14)

    Returns:
      Pandas Series with RSI values.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi