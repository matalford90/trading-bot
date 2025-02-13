class RiskManager:
    def __init__(self, total_capital, max_allocation=0.5, max_trades=5):
        self.total_capital = total_capital
        self.max_allocation = max_allocation  # Use only 50% of the account
        self.max_trades = max_trades

    def allocation_per_trade(self):
        return (self.total_capital * self.max_allocation) / self.max_trades

    def calculate_position_size(self, risk_fraction, stop_loss_distance, entry_price):
        # For example, risk 2% of allocated capital per trade.
        risk_amount = self.allocation_per_trade() * risk_fraction
        return risk_amount / stop_loss_distance

    def adjust_leverage(self, signal_confidence):
        base_leverage = 1
        leverage = base_leverage + signal_confidence * 4
        return min(leverage, 10)