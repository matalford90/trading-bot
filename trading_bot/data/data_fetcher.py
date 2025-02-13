import time
import requests
import sys
import os
import pandas as pd  # Import pandas for CSV handling

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import KUCOIN_API_KEY, KUCOIN_SECRET, KUCOIN_PASSPHRASE
from ..utils.logger import setup_logger

logger = setup_logger('data_fetcher', 'logs/data_fetcher.log')

class KucoinDataFetcher:
    def __init__(self):
        self.base_url = 'https://api-futures.kucoin.com'
<<<<<<< HEAD
    
    def get_candlesticks(self, symbol, granularity=60, total_limit=5000):
        """
        Fetch a large amount of historical candlestick data using pagination.
        """
        all_candles = []
        batch_size = 1000  # API usually limits responses to 1000 candles per request
        end_time_ms = int(time.time() * 1000)  # Current timestamp in milliseconds

        while len(all_candles) < total_limit:
            start_time_ms = end_time_ms - (batch_size * granularity * 1000)

            params = {
                'symbol': symbol,
                'granularity': granularity,
                'from': start_time_ms,
                'to': end_time_ms
            }
            url = self.base_url + '/api/v1/kline/query'
            response = requests.get(url, params=params)
        
            if response.status_code == 200:
                new_candles = response.json().get('data', [])
                if not new_candles:
                    break  # Stop if no new candles are returned

                all_candles.extend(new_candles)
                end_time_ms = new_candles[0][0]  # Move the end time backward
            else:
                print(f"❌ Error fetching market data: {response.status_code}")
                break

        return all_candles[:total_limit]  # Return only the requested amount

def aggregate_candles(candles, target_interval):
    """
    Aggregate 1-minute candles into candles of target_interval minutes.
    """
    aggregated = []
    candles = sorted(candles, key=lambda x: x[0])
=======

    def get_candlesticks(self, symbol, granularity=60, limit=50):
        """
        Fetch candlestick (k-line) data from KuCoin Futures.
        """
        endpoint = '/api/v1/kline/query'
        current_time_ms = int(time.time() * 1000)
        start_time_ms = current_time_ms - limit * 60 * 1000  # 1 minute = 60,000 ms
        params = {
            'symbol': symbol,
            'granularity': granularity,
            'from': start_time_ms,
            'to': current_time_ms
        }
        url = self.base_url + endpoint

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get('code') == '200000':
                logger.info(f"Fetched {len(data.get('data', []))} candlesticks for {symbol}")
                return data['data']
            else:
                logger.error(f"API error: {data}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error fetching candlesticks: {e}")
            return None

def aggregate_candles(candles, target_interval):
    """
    Aggregate 1-minute candles into target_interval minute candles.
    """
    aggregated = []
    candles = sorted(candles, key=lambda x: x[0])

>>>>>>> 7d64dceb (Updated feature XYZ)
    for i in range(0, len(candles), target_interval):
        group = candles[i:i+target_interval]
        if not group:
            continue
        start_time = group[0][0]
        open_price = float(group[0][1])
        high_price = max(float(c[2]) for c in group)
        low_price = min(float(c[3]) for c in group)
        close_price = float(group[-1][4])
        total_volume = sum(float(c[5]) for c in group)
        aggregated.append([start_time, open_price, high_price, low_price, close_price, total_volume])

    logger.info(f"Aggregated {len(aggregated)} candles into {target_interval}-minute intervals")
    return aggregated

def save_to_csv(data, filename="trading_bot/data/market_data.csv"):
    """
    Save market data to CSV.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists

    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")  # Convert timestamp

    df.to_csv(filename, index=False)
    print(f"✅ Market data saved to {filename}")

if __name__ == '__main__':
    fetcher = KucoinDataFetcher()
<<<<<<< HEAD
    
    # Fetch 1-minute data using symbol 'XBTUSDM'
    raw_data = fetcher.get_candlesticks('XBTUSDM', granularity=60, total_limit=1000)

    print(f"✅ Fetched {len(raw_data)} raw candles.")
    print(f"📊 Sample candle: {raw_data[0]}")  # Print first candle to check format

    if isinstance(raw_data, list) and len(raw_data) > 0:
        candles = raw_data
        print(f"✅ Successfully fetched {len(candles)} raw candles.")

        if len(candles) < 50:
            print("⚠️ Warning: Not enough data fetched. API might be limiting the request.")

        # Aggregate and save data
        aggregated = aggregate_candles(candles, target_interval=15)
        print(f"✅ Aggregated {len(aggregated)} 15-minute candles.")
        save_to_csv(aggregated)
    else:
        print("❌ Error fetching 1-minute candles:", raw_data)
=======
    raw_data = fetcher.get_candlesticks('XBTUSDM', granularity=60, limit=50)

    if raw_data:
        print("Raw 1-minute candles:", raw_data)
        aggregated = aggregate_candles(raw_data, target_interval=15)
        print("\nAggregated 15-minute candles:", aggregated)
    else:
        logger.error("Failed to fetch candlestick data")
>>>>>>> 7d64dceb (Updated feature XYZ)
