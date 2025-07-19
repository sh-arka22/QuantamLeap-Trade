# File: backend/agents/base_agent.py
class BaseAgent:
    def __init__(self, trend, initial_money=10000):
        self.trend = trend
        self.initial_money = initial_money

    def buy(self):
        raise NotImplementedError("Buy method must be implemented by subclasses.")
