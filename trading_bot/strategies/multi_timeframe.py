def aggregate_timeframe_signals(signal_15m, signal_1h, signal_4h):
    # For example, only proceed if the 15m signal aligns with the trend of 1h and 4h charts.
    if signal_15m['signal'] != 'none' and \
       signal_15m['signal'] == signal_1h.get('signal') and \
       signal_15m['signal'] == signal_4h.get('signal'):
        # Boost confidence if all timeframes agree.
        return {'signal': signal_15m['signal'], 'confidence': min(signal_15m['confidence'] + 0.3, 1.0)}
    return signal_15m  # Otherwise, default to the 15-minute signal.