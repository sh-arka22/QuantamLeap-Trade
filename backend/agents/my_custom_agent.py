# File: backend/agents/my_custom_agent.py
from .signal_rolling import SignalRollingAgent
from .moving_average import MovingAverageAgent
from .abcd_strategy import ABCDStrategyAgent
from .evolution_strategy import EvolutionStrategyAgent
from .cnn_lstm import CNNLSTMAgent
from ..utils.data_utils import load_stock_data

class MyCustomAgent:
    mapping = {
        "Evolution Agent": EvolutionStrategyAgent,
        "MA Crossover": MovingAverageAgent,
        "ABCD Strategy": ABCDStrategyAgent,
        "CNN-LSTM": CNNLSTMAgent,
        "Signal Rolling": SignalRollingAgent
    }
    
    def __init__(self, trend, initial_money=10000, agent_name="Evolution Agent", **kw):
        AgentCls = self.mapping.get(agent_name)
        if not AgentCls:
            raise ValueError(f"Unknown strategy {agent_name}")
        self.agent = AgentCls(trend, initial_money, **kw)

    def buy(self):
        return self.agent.buy()
    
    def predict(self, data):
        if hasattr(self.agent, "predict"):
            return self.agent.predict(data)
        else:
            raise NotImplementedError(f"The agent {type(self.agent).__name__} does not implement predict().")
    
    def train(self, data):
        if hasattr(self.agent, "train"):
            return self.agent.train(data)
        else:
            raise NotImplementedError(f"The agent {type(self.agent).__name__} does not implement train().")
