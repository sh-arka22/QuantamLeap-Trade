# File: backend/agents/moving_average.py
from .base_agent import BaseAgent
import pandas as pd
import numpy as np

class MovingAverageAgent(BaseAgent):
    def buy(self, short_window=None, long_window=None):
        if short_window is None:
            short_window = int(0.025 * len(self.trend))
        if long_window is None:
            long_window = int(0.05 * len(self.trend))

        df = pd.DataFrame({'Close': self.trend})
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0.0
        signals['short_ma'] = df['Close'].rolling(window=short_window, min_periods=1).mean()
        signals['long_ma'] = df['Close'].rolling(window=long_window, min_periods=1).mean()
        signals['signal'][short_window:] = np.where(
            signals['short_ma'][short_window:] > signals['long_ma'][short_window:], 1.0, 0.0
        )
        signals['positions'] = signals['signal'].diff()

        states_buy = signals[signals['positions'] == 1.0].index.tolist()
        states_sell = signals[signals['positions'] == -1.0].index.tolist()

        initial_money = self.initial_money
        for t in states_buy:
            if initial_money >= df['Close'].iloc[t]:
                initial_money -= df['Close'].iloc[t]
        for t in states_sell:
            initial_money += df['Close'].iloc[t]

        invest = ((initial_money - self.initial_money) / self.initial_money) * 100
        total_gains = initial_money - self.initial_money
        return states_buy, states_sell, total_gains, invest
