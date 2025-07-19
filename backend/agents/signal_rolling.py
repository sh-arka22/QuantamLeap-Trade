# File: backend/agents/signal_rolling.py
from .base_agent import BaseAgent
import numpy as np

class SignalRollingAgent(BaseAgent):
    def buy(self, delay=5, max_buy=1, max_sell=1):
        real_movement = np.array(self.trend)
        starting_money = self.initial_money
        current_decision = 0
        state = 1
        current_val = real_movement[0]
        states_sell = []
        states_buy = []
        current_inventory = 0

        def buy_action(i, money, inventory):
            shares = money // real_movement[i]
            if shares < 1:
                return money, inventory
            buy_units = min(shares, max_buy)
            money -= buy_units * real_movement[i]
            inventory += buy_units
            states_buy.append(i)
            return money, inventory

        if state == 1:
            self.initial_money, current_inventory = buy_action(0, self.initial_money, current_inventory)

        for i in range(1, len(real_movement)):
            if real_movement[i] < current_val and state == 0:
                if current_decision < delay:
                    current_decision += 1
                else:
                    state = 1
                    self.initial_money, current_inventory = buy_action(i, self.initial_money, current_inventory)
                    current_decision = 0
            if real_movement[i] > current_val and state == 1:
                if current_decision < delay:
                    current_decision += 1
                else:
                    state = 0
                    if current_inventory > 0:
                        sell_units = min(current_inventory, max_sell)
                        current_inventory -= sell_units
                        total_sell = sell_units * real_movement[i]
                        self.initial_money += total_sell
                        states_sell.append(i)
                    current_decision = 0
            current_val = real_movement[i]

        invest = ((self.initial_money - starting_money) / starting_money) * 100
        return states_buy, states_sell, self.initial_money - starting_money, invest
