def evaluate_signals(signal_list):
    final_signal = {'signal': 'none', 'confidence': 0.0}
    for sig in signal_list:
        if sig['signal'] != 'none' and sig['confidence'] > final_signal['confidence']:
            final_signal = sig
    return final_signal