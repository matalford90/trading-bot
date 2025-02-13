print("✅ Order Manager script started...")

import json
import requests
import time
import hmac
import hashlib
import base64
import uuid  # Generates a unique client order ID
import sys
import os
from decimal import Decimal, ROUND_DOWN  # Fix floating-point rounding issues

# Get the absolute path of the project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from trading_bot.config import KUCOIN_API_KEY, KUCOIN_SECRET, KUCOIN_PASSPHRASE
<<<<<<< HEAD
from trading_bot.risk.risk_manager import RiskManager  # Import RiskManager
=======
from trading_bot.utils.logger import setup_logger  # Import logging utility

# Setup logger
logger = setup_logger('order_manager', 'logs/order_manager.log')
>>>>>>> 7d64dceb (Updated feature XYZ)

class OrderManager:
    def __init__(self, total_capital=10000):  # Added total capital for risk management
        self.base_url = 'https://api-futures.kucoin.com'
        self.api_key = KUCOIN_API_KEY
        self.api_secret = KUCOIN_SECRET
        self.passphrase = KUCOIN_PASSPHRASE

        # Initialize RiskManager
        self.risk_manager = RiskManager(total_capital=total_capital)

    def _generate_signature(self, method, endpoint, body=""):
        """
        Generates the authentication signature required for KuCoin API.
        """
        now = str(int(time.time() * 1000))  # Current timestamp in milliseconds
        str_to_sign = now + method + endpoint + body
        signature = base64.b64encode(hmac.new(
            self.api_secret.encode(), str_to_sign.encode(), hashlib.sha256
        ).digest()).decode()

        passphrase_signature = base64.b64encode(hmac.new(
            self.api_secret.encode(), self.passphrase.encode(), hashlib.sha256
        ).digest()).decode()

        return now, signature, passphrase_signature

    def get_contract_details(self, symbol):
        """
        Fetch contract details including minimum price, leverage, and tick size requirements.
        """
        endpoint = f"/api/v1/contracts/{symbol}"
        url = self.base_url + endpoint
        try:
            response = requests.get(url)
            data = response.json()

            if response.status_code == 200 and data.get("code") == "200000":
                logger.info(f"✅ Successfully fetched contract details for {symbol}")
                return data["data"]
            else:
                logger.error(f"❌ API error fetching contract details: {data}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"🚨 Request error fetching contract details: {e}")
            return None

    def _round_price_to_tick_size(self, price, tick_size):
        """
        Ensures the price is a valid multiple of tick size using Decimal for precision.
        """
        tick_size = Decimal(str(tick_size))
        price = Decimal(str(price))
        rounded_price = (price / tick_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * tick_size
        return float(rounded_price)

    def place_order(self, symbol, side, risk_fraction=0.02, stop_loss_distance=1000, order_type="market", price=None):
        """
        Places an order on KuCoin Futures with risk management.
        - risk_fraction: Fraction of allocated capital to risk per trade (e.g., 0.02 for 2% risk).
        - stop_loss_distance: The price difference between entry and stop-loss.
        """

        contract_details = self.get_contract_details(symbol)
        if not contract_details:
            logger.error(f"❌ Failed to fetch contract details for {symbol}. Order not placed.")
            return None

<<<<<<< HEAD
        # Get dynamic position size based on risk
        mark_price = float(contract_details.get("markPrice", 0))  # Get current market price
        position_size = self.risk_manager.calculate_position_size(risk_fraction, stop_loss_distance, mark_price)
        position_size = max(1, round(position_size))  # Ensure minimum trade size

        # Adjust leverage based on confidence (assuming signal confidence = 0.5 for now)
        leverage = self.risk_manager.adjust_leverage(0.5)

        print(f"📊 Calculated Position Size: {position_size}, Adjusted Leverage: {leverage}")
=======
        max_leverage = contract_details.get("maxLeverage", 10)
        if leverage > max_leverage:
            logger.warning(f"⚠️ Adjusting leverage from {leverage} to max allowed: {max_leverage}")
            leverage = max_leverage
>>>>>>> 7d64dceb (Updated feature XYZ)

        tick_size = float(contract_details.get("tickSize", 0.1))

        if order_type == "limit":
            sell_limit = float(contract_details.get("sellLimit", 0))
            buy_limit = float(contract_details.get("buyLimit", 0))

<<<<<<< HEAD
            if price is None:  # Auto-set price if not provided
                if side == "buy":
                    price = buy_limit
                    print(f"⚠️ Adjusting buy limit price to: {buy_limit}")
                else:
                    price = sell_limit
                    print(f"⚠️ Adjusting sell limit price to: {sell_limit}")

            if side == "buy" and float(price) < buy_limit:
                print(f"⚠️ Adjusting buy order price to minimum allowed: {buy_limit}")
                price = buy_limit

            if side == "sell":
                min_sell_price = float(contract_details.get("sellLimit", 0))
                if price is None or float(price) < min_sell_price:
                    print(f"⚠️ Adjusting sell order price to KuCoin's minimum: {min_sell_price}")
                    price = min_sell_price

                # Round price to the nearest tick size multiple (avoiding floating-point errors)
                price = self._round_price_to_tick_size(price, tick_size)
                print(f"🔄 Rounded sell price to nearest tick size ({tick_size}): {price}")
=======
            if price is None:
                price = buy_limit if side == "buy" else sell_limit
                logger.warning(f"⚠️ Auto-adjusted {side} limit order price to: {price}")
            elif side == "buy" and float(price) < buy_limit:
                logger.warning(f"⚠️ Adjusted buy price to minimum allowed: {buy_limit}")
                price = buy_limit
            elif side == "sell":
                if float(price) < sell_limit:
                    logger.warning(f"⚠️ Adjusted sell order price to minimum allowed: {sell_limit}")
                    price = sell_limit
                elif float(price) < contract_details.get("minPrice", sell_limit):
                    min_price = contract_details.get("minPrice", sell_limit)
                    logger.warning(f"⚠️ Adjusted sell order price to KuCoin min price: {min_price}")
                    price = min_price

            price = self._round_price_to_tick_size(price, tick_size)
            logger.info(f"🔄 Rounded price to nearest tick size ({tick_size}): {price}")
>>>>>>> 7d64dceb (Updated feature XYZ)

        endpoint = '/api/v1/orders'
        url = self.base_url + endpoint
        method = 'POST'

        order_data = {
            'clientOid': str(uuid.uuid4()),
            'symbol': symbol,
<<<<<<< HEAD
            'side': side,  # 'buy' or 'sell'
            'size': position_size,  # Use calculated position size
=======
            'side': side,
            'size': size,
>>>>>>> 7d64dceb (Updated feature XYZ)
            'leverage': leverage,
            'type': order_type,
        }

        if order_type == "limit":
            order_data['price'] = price

        body = json.dumps(order_data)
        timestamp, signature, passphrase_signature = self._generate_signature(method, endpoint, body)

        headers = {
            "KC-API-KEY": self.api_key,
            "KC-API-SIGN": signature,
            "KC-API-TIMESTAMP": timestamp,
            "KC-API-PASSPHRASE": passphrase_signature,
            "KC-API-KEY-VERSION": "2",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(url, headers=headers, json=order_data)
            response_data = response.json()

            if response.status_code == 200 and response_data.get("code") == "200000":
                logger.info(f"✅ Order placed successfully: {response_data}")
                return response_data
            else:
                logger.error(f"❌ Error placing order: {response_data}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"🚨 Request error placing order: {e}")
            return None

# Example Usage
if __name__ == "__main__":
    order_manager = OrderManager()

<<<<<<< HEAD
    # Market Order Example
    result_market = order_manager.place_order(
        symbol="XBTUSDTM",  
        side="buy",
        risk_fraction=0.02,  
        stop_loss_distance=500,  
        order_type="market"
    )
    print(result_market)

    # Limit Order Example with Auto-Adjusted Price
    result_limit = order_manager.place_order(
        symbol="XBTUSDTM",
        side="sell",
        risk_fraction=0.02,
        stop_loss_distance=500,
        order_type="limit",
        price=None  
    )
    print(result_limit)
=======
    try:
        # Market Order Example
        result_market = order_manager.place_order(
            symbol="XBTUSDTM",
            side="buy",
            size=1,
            leverage=10,
            order_type="market"
        )
        logger.info(f"Market Order Result: {result_market}")
        print(f"Market Order Result: {result_market}")

        # Limit Order Example with Auto-Adjusted Price
        result_limit = order_manager.place_order(
            symbol="XBTUSDTM",
            side="sell",
            size=1,
            leverage=10,
            order_type="limit",
            price=None
        )
        logger.info(f"Limit Order Result: {result_limit}")
        print(f"Limit Order Result: {result_limit}")

    except Exception as e:
        logger.error(f"🚨 Fatal error in order execution: {e}")
        print(f"🚨 Fatal error: {e}")

    logger.info("✅ Order Manager script executed.")
    print("✅ Order Manager script executed.")  # Force print to terminal

    print("⏳ Waiting 5 seconds before exit...")
    time.sleep(5)
>>>>>>> 7d64dceb (Updated feature XYZ)
