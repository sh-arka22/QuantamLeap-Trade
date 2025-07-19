# File: backend/agents/evolution_strategy.py
from .base_agent import BaseAgent
import numpy as np

class DeepEvolutionStrategy:
    def __init__(self, weights, reward_function, population_size, sigma, learning_rate):
        self.weights = weights
        self.reward_function = reward_function
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    def _get_weight_from_population(self, weights, population):
        return [weights[index] + self.sigma * i for index, i in enumerate(population)]

    def get_weights(self):
        return self.weights

    def train(self, epoch=100):
        for _ in range(epoch):
            population = [[np.random.randn(*w.shape) for w in self.weights] for _ in range(self.population_size)]
            rewards = np.array([self.reward_function(self._get_weight_from_population(self.weights, p)) for p in population])
            rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-7)
            for index, w in enumerate(self.weights):
                A = np.array([p[index] for p in population])
                self.weights[index] = w + self.learning_rate / (self.population_size * self.sigma) * np.dot(A.T, rewards).T

class EvolutionModel:
    def __init__(self, input_size, layer_size, output_size):
        self.weights = [
            np.random.randn(input_size, layer_size),
            np.random.randn(layer_size, output_size),
            np.random.randn(1, layer_size),
        ]

    def predict(self, inputs):
        feed = np.dot(inputs, self.weights[0]) + self.weights[-1]
        return np.dot(feed, self.weights[1])

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

class EvolutionStrategyAgent(BaseAgent):
    def __init__(self, trend, initial_money=10000, window_size=30, skip=1):
        super().__init__(trend, initial_money)
        self.window_size = window_size
        self.skip = skip
        self.model = EvolutionModel(window_size, 500, 3)
        self.es = DeepEvolutionStrategy(
            self.model.get_weights(),
            self.get_reward,
            population_size=15,
            sigma=0.1,
            learning_rate=0.03
        )

    def get_state(self, t):
        window_size = self.window_size + 1
        d = t - window_size + 1
        block = self.trend[d:t+1] if d >= 0 else -d * [self.trend[0]] + self.trend[0:t+1]
        return np.array([[block[i+1] - block[i] for i in range(window_size - 1)]])

    def act(self, sequence):
        return np.argmax(self.model.predict(np.array(sequence))[0])

    def get_reward(self, weights):
        self.model.set_weights(weights)
        initial_money = self.initial_money
        starting_money = initial_money
        inventory = []
        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(self.get_state(t))
            if action == 1 and starting_money >= self.trend[t]:
                inventory.append(self.trend[t])
                starting_money -= self.trend[t]
            elif action == 2 and inventory:
                starting_money += self.trend[t]
                inventory.pop(0)
        return ((starting_money - initial_money) / initial_money) * 100

    def fit(self, iterations=500):
        self.es.train(epoch=iterations)

    def buy(self):
        self.fit()
        initial_money = self.initial_money
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(self.get_state(t))
            if action == 1 and initial_money >= self.trend[t]:
                inventory.append(self.trend[t])
                initial_money -= self.trend[t]
                states_buy.append(t)
            elif action == 2 and inventory:
                initial_money += self.trend[t]
                states_sell.append(t)
                inventory.pop(0)
        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        return states_buy, states_sell, total_gains, invest
