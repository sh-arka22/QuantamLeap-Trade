# File: backend/portfolio.py
class Portfolio:
    """
    Simple portfolio class to track balance and trades.
    """
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades = []  # list of trades

    def record_trade(self, action, price, amount):
        self.trades.append({"action": action, "price": price, "amount": amount})
        if action == "buy":
            self.balance -= price * amount
        elif action == "sell":
            self.balance += price * amount

    def get_balance(self):
        return self.balance

    def get_profit(self):
        return self.balance - self.initial_balance

    def get_trades(self):
        return self.trades
