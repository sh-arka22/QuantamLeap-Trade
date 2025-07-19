# File: backend/agents/cnn_lstm.py
from .base_agent import BaseAgent
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense

class CNNLSTMAgent(BaseAgent):
    def __init__(self, trend, initial_money=10000, window_size=30):
        super().__init__(trend, initial_money)
        self.window_size = window_size
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(self.window_size, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def get_state(self, t):
        d = t - self.window_size + 1
        block = self.trend[d:t+1] if d >= 0 else -d * [self.trend[0]] + self.trend[0:t+1]
        return np.array([block]).reshape(1, self.window_size, 1)

    def train(self, epochs=100):
        X = [self.get_state(t) for t in range(self.window_size, len(self.trend)-1)]
        y = self.trend[self.window_size+1:]
        self.model.fit(np.vstack(X), np.array(y), epochs=epochs, verbose=0)

    def predict(self, state):
        return self.model.predict(state)[0][0]

    def buy(self):
        self.train()
        initial_money = self.initial_money
        state = self.get_state(0)
        inventory = []
        states_buy = []
        states_sell = []
        for t in range(0, len(self.trend) - 1, 5):
            action = 'buy' if self.predict(state) > self.trend[t] else 'sell'
            next_state = self.get_state(t + 1)
            if action == 'buy' and initial_money >= self.trend[t]:
                inventory.append(self.trend[t])
                initial_money -= self.trend[t]
                states_buy.append(t)
            elif action == 'sell' and inventory:
                inventory.pop(0)
                initial_money += self.trend[t]
                states_sell.append(t)
            state = next_state
        invest = ((initial_money - self.initial_money) / self.initial_money) * 100
        return states_buy, states_sell, initial_money - self.initial_money, invest
