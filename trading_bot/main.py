import time
import torch
from data.data_fetcher import KucoinDataFetcher
from data_processing import compute_indicators  # Corrected import path
from strategies.divergence_strategy import DivergenceStrategy
from strategies.sr_strategy import SRStrategy
from strategies.multi_timeframe import aggregate_timeframe_signals
from strategies.signal_evaluator import evaluate_signals
from risk.risk_manager import RiskManager
from execution.order_manager import OrderManager
from models.ml_model import TradingModel
from models.trainer import train_model, TradingDataset
from social.social_monitor import SocialMonitor
from utils.logger import logger
from config import TOTAL_CAPITAL, CHECK_INTERVAL, ML_INPUT_SIZE

# Define futures trading pairs (this list can be dynamically updated)
TRADING_PAIRS = ['BTCUSDTM', 'ETHUSDTM', 'ADAUSDTM', 'XRPUSDTM', 'SOLUSDTM', 'DOTUSDTM']

# Containers for online learning data
online_features = []
online_labels = []

def main():
    fetcher = KucoinDataFetcher()
    risk_manager = RiskManager(total_capital=TOTAL_CAPITAL)
    order_manager = OrderManager()
    social_monitor = SocialMonitor()

    # Initialize the ML model.
    model = TradingModel(input_size=ML_INPUT_SIZE, hidden_size=32, output_size=1)
    # Optionally load pre-trained weights:
    # model.load_state_dict(torch.load('model_weights.pth'))

    while True:
        pair_signals = []
        for symbol in TRADING_PAIRS:
            # --- 15-minute timeframe analysis ---
            candles_15m = fetcher.get_candlesticks(symbol, granularity=15)
            data_15m = compute_indicators(candles_15m)
            div_signal_15m = DivergenceStrategy(data_15m).generate_signal()
            sr_signal_15m = SRStrategy(data_15m).generate_signal()
            signal_15m = evaluate_signals([div_signal_15m, sr_signal_15m])
            
            # --- 1-hour timeframe analysis ---
            candles_1h = fetcher.get_candlesticks(symbol, granularity=60)
            data_1h = compute_indicators(candles_1h)
            div_signal_1h = DivergenceStrategy(data_1h).generate_signal()
            sr_signal_1h = SRStrategy(data_1h).generate_signal()
            signal_1h = evaluate_signals([div_signal_1h, sr_signal_1h])
            
            # --- 4-hour timeframe analysis ---
            candles_4h = fetcher.get_candlesticks(symbol, granularity=240)
            data_4h = compute_indicators(candles_4h)
            div_signal_4h = DivergenceStrategy(data_4h).generate_signal()
            sr_signal_4h = SRStrategy(data_4h).generate_signal()
            signal_4h = evaluate_signals([div_signal_4h, sr_signal_4h])
            
            # Aggregate signals across multiple timeframes.
            aggregated_signal = aggregate_timeframe_signals(signal_15m, signal_1h, signal_4h)
            aggregated_signal['symbol'] = symbol
            pair_signals.append(aggregated_signal)
        
        # --- Social Sentiment Integration ---
        news_articles = social_monitor.get_crypto_news(query="crypto")
        sentiment_score = social_monitor.analyze_sentiment(news_articles)
        logger.info(f"Social sentiment score: {sentiment_score}")

        # Adjust signals based on social sentiment.
        for sig in pair_signals:
            if sentiment_score > 0 and sig['signal'] == 'buy':
                sig['confidence'] = min(sig['confidence'] + 0.1, 1.0)
            elif sentiment_score < 0 and sig['signal'] == 'sell':
                sig['confidence'] = min(sig['confidence'] + 0.1, 1.0)

        # --- Select Top 5 Trading Opportunities ---
        pair_signals = sorted(pair_signals, key=lambda x: x['confidence'], reverse=True)
        active_trades = [s for s in pair_signals if s['signal'] != 'none'][:5]
        
        for trade in active_trades:
            symbol = trade['symbol']
            signal = trade['signal']
            confidence = trade['confidence']
            # Retrieve the current market price (replace placeholder with actual API call)
            entry_price = 100
            stop_loss_distance = 50
            position_size = risk_manager.calculate_position_size(0.02, stop_loss_distance, entry_price)
            leverage = risk_manager.adjust_leverage(confidence)
            logger.info(f"Placing {signal} order on {symbol} with size {position_size} and {leverage}x leverage.")
            order_response = order_manager.place_order(symbol, signal, position_size, leverage)
            logger.info(f"Order response: {order_response}")
            
            # --- Collect Data for Online Learning ---
            features = [0.0] * ML_INPUT_SIZE  # Replace with real feature extraction logic.
            label = 0.0                        # Replace with actual outcome (price change, PNL, etc.)
            online_features.append(features)
            online_labels.append(label)
        
        # --- Online Model Update ---
        if len(online_features) >= 32:
            dataset = TradingDataset(torch.tensor(online_features, dtype=torch.float32),
                                     torch.tensor(online_labels, dtype=torch.float32))
            train_model(model, dataset, num_epochs=5)
            # Optionally save updated model weights:
            # torch.save(model.state_dict(), 'model_weights.pth')
            online_features.clear()
            online_labels.clear()

        # Optionally, update a backend data store for mobile API queries.
        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    main()