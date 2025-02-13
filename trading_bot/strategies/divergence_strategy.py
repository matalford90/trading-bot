from data.data_processor import detect_divergence

class DivergenceStrategy:
    def __init__(self, data):
        self.data = data

    def generate_signal(self):
        signals = detect_divergence(self.data)
        if signals.get('bullish_divergence'):
            return {'signal': 'buy', 'confidence': 0.7}
        elif signals.get('bearish_divergence'):
            return {'signal': 'sell', 'confidence': 0.7}
        return {'signal': 'none', 'confidence': 0.0}