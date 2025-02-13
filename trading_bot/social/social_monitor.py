import requests
import logging
from textblob import TextBlob
import sys
import os
import csv
import datetime
import sqlite3

# Ensure the script can find config.py inside trading_bot/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CMC_API_KEY  # Using CoinMarketCap API Key

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class SocialMonitor:
    def __init__(self):
        self.news_api_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"  # CoinMarketCap API URL

    def get_crypto_news(self, limit=10):
        """Fetch latest cryptocurrency listings from CoinMarketCap."""
        headers = {
            "Accepts": "application/json",
            "X-CMC_PRO_API_KEY": CMC_API_KEY  # API key in headers
        }
        params = {
            "limit": limit,  # Fetch latest 10 listings
            "sort": "market_cap",  # Sort by market cap
            "convert": "USD"
        }
        try:
            response = requests.get(self.news_api_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if "data" not in data:
                logging.warning("❌ API response does not contain expected data format.")
                return []

            articles = data["data"]  # Extract cryptocurrency data
            logging.info(f"✅ Fetched {len(articles)} cryptocurrency listings from CoinMarketCap")

            return articles
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Error fetching news from CoinMarketCap: {e}")
            return []

    def analyze_sentiment(self, articles):
        """Perform sentiment analysis on cryptocurrency names and tags."""
        total_score = 0.0
        analyzed_count = 0

        sentiment_weights = {
            "mineable": 0.2, "store-of-value": 0.5, "smart-contracts": 0.3,
            "defi": 0.4, "scam": -0.8, "hacked": -0.9, "rug-pull": -1.0,
            "bullish": 0.6, "bearish": -0.6, "pump": 0.7, "dump": -0.7,
            "volatile": -0.3, "stablecoin": 0.2
        }

        for article in articles:
            sentiment_text = ""

            if "name" in article and article["name"]:
                sentiment_text += article["name"] + " "

            if "tags" in article and isinstance(article["tags"], list):
                sentiment_text += " ".join(article["tags"]) + " "

            if sentiment_text.strip():
                analysis = TextBlob(sentiment_text)
                sentiment_score = analysis.sentiment.polarity  

                for word, weight in sentiment_weights.items():
                    if word in sentiment_text.lower():
                        sentiment_score += weight

                total_score += sentiment_score
                analyzed_count += 1

        avg_sentiment = total_score / analyzed_count if analyzed_count > 0 else 0
        logging.info(f"📊 Analyzed {analyzed_count} listings. Average Sentiment Score: {avg_sentiment:.2f}")
        return avg_sentiment

    def store_data_csv(self, timestamp, sentiment_score, coin_name, coin_price, market_cap):
        """Save sentiment analysis data to a CSV file."""
        csv_filename = "crypto_sentiment_data.csv"
        csv_headers = ["Timestamp", "Sentiment Score", "Top Coin", "Price (USD)", "Market Cap (USD)"]

        file_exists = os.path.isfile(csv_filename)
        try:
            with open(csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(csv_headers)  # Write header only once
                writer.writerow([timestamp, sentiment_score, coin_name, coin_price, market_cap])
            logging.info(f"📁 Data saved to {csv_filename}")
        except Exception as e:
            logging.error(f"❌ Failed to write to CSV: {e}")

    def store_data_db(self, timestamp, sentiment_score, coin_name, coin_price, market_cap):
        """Save sentiment data to an SQLite database."""
        try:
            conn = sqlite3.connect("crypto_sentiment.db")
            with conn:
                cursor = conn.cursor()

                # Create table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sentiment_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        sentiment_score REAL,
                        top_coin TEXT,
                        price REAL,
                        market_cap REAL
                    )
                """)

                # Insert data
                cursor.execute("INSERT INTO sentiment_data (timestamp, sentiment_score, top_coin, price, market_cap) VALUES (?, ?, ?, ?, ?)",
                            (timestamp, sentiment_score, coin_name, coin_price, market_cap))

                conn.commit()
                logging.info(f"✅ Data saved to database: crypto_sentiment.db")
        except sqlite3.Error as e:
            logging.error(f"❌ Database error: {e}")

if __name__ == '__main__':
    monitor = SocialMonitor()
    crypto_news = monitor.get_crypto_news()

    if crypto_news:
        sentiment_score = monitor.analyze_sentiment(crypto_news)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        first_coin = crypto_news[0] if crypto_news else None
        if first_coin and "quote" in first_coin and "USD" in first_coin["quote"]:
            coin_name = first_coin.get("name", "Unknown")
            coin_price = first_coin["quote"]["USD"].get("price", 0)
            market_cap = first_coin["quote"]["USD"].get("market_cap", 0)

            monitor.store_data_csv(timestamp, sentiment_score, coin_name, coin_price, market_cap)
            monitor.store_data_db(timestamp, sentiment_score, coin_name, coin_price, market_cap)

            print(f"📊 Crypto Market Sentiment Score: {sentiment_score:.2f} | Data Stored Successfully")
        else:
            logging.warning("⚠️ No valid cryptocurrency data found.")