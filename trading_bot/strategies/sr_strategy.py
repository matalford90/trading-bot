from data.data_processor import detect_support_resistance

class SRStrategy:
    def __init__(self, data):
        self.data = data

    def generate_signal(self):
        signals = detect_support_resistance(self.data)
        if signals.get('touching_support'):
            return {'signal': 'buy', 'confidence': 0.6}
        elif signals.get('touching_resistance'):
            return {'signal': 'sell', 'confidence': 0.6}
        return {'signal': 'none', 'confidence': 0.0}