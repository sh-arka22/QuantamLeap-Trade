# File: backend/agents/abcd_strategy.py
from .base_agent import BaseAgent
import pandas as pd
import numpy as np

class ABCDStrategyAgent(BaseAgent):
    def abcd(self, ma_window=7, skip_loop=4):
        ma = pd.Series(self.trend).rolling(ma_window).mean().values
        x = []
        for a in range(len(ma)):
            for b in range(a, len(ma), skip_loop):
                for c in range(b, len(ma), skip_loop):
                    for d in range(c, len(ma), skip_loop):
                        if ma[b] > ma[a] and (ma[c] < ma[b] and ma[c] > ma[a]) and ma[d] > ma[b]:
                            x.append([a, b, c, d])
        if not x:
            return np.zeros(len(self.trend))
        x_np = np.array(x)
        ac = x_np[:, 0].tolist() + x_np[:, 2].tolist()
        bd = x_np[:, 1].tolist() + x_np[:, 3].tolist()
        ac_set = set(ac)
        bd_set = set(bd)
        signal = np.zeros(len(self.trend))
        buy = list(ac_set - bd_set)
        sell = list(bd_set - ac_set)
        signal[buy] = 1.0
        signal[sell] = -1.0
        return signal

    def buy(self, ma=7, skip_loop=4, max_buy=1, max_sell=1):
        signal = self.abcd(ma, skip_loop)
        initial_money = self.initial_money
        states_buy = []
        states_sell = []
        current_inventory = 0
        for i in range(len(self.trend)):
            state = signal[i]
            if state == 1:
                shares = initial_money // self.trend[i]
                if shares >= 1:
                    buy_units = min(shares, max_buy)
                    initial_money -= buy_units * self.trend[i]
                    current_inventory += buy_units
                    states_buy.append(i)
            elif state == -1 and current_inventory > 0:
                sell_units = min(current_inventory, max_sell)
                current_inventory -= sell_units
                total_sell = sell_units * self.trend[i]
                initial_money += total_sell
                states_sell.append(i)
        invest = ((initial_money - self.initial_money) / self.initial_money) * 100
        total_gains = initial_money - self.initial_money
        return states_buy, states_sell, total_gains, invest
