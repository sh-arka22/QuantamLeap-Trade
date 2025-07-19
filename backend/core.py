# File: backend/core.py
from datetime import datetime
import os
import numpy as np
import pandas as pd
import logging
from backend.models.cnn_lstm_model import CNNLSTMModel
from backend.agents.my_custom_agent import MyCustomAgent
from backend.utils.data_utils import load_stock_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global state for portfolio
portfolio_value = 10000.0  # initial balance
total_profit = 0.0
win_rate = 0.0  # dummy

def list_csv_files():
    """
    Return a list of stock symbols (CSV filenames minus the .csv).
    """
    data_dir = os.path.join(os.getcwd(), "backend", "data")
    symbols = []
    try:
        for f in os.listdir(data_dir):
            if f.lower().endswith(".csv"):
                symbol, _ = os.path.splitext(f)
                symbols.append(symbol)
    except FileNotFoundError:
        pass
    return symbols



def upload_csv(file_storage):
    data_dir = os.path.join(os.getcwd(), "backend", "data")
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, file_storage.filename)
    file_storage.save(filepath)
    return {"message": f"File {file_storage.filename} uploaded successfully."}

def get_predictions(symbol, days=30):
    """Return a naive price forecast for ``symbol``.

    If ``symbol`` is missing or the CSV is unavailable, an error dict is
    returned rather than raising an exception so the API can gracefully
    handle invalid requests.
    """
    if not symbol:
        return {"error": "Symbol not provided."}

    # Allow passing either ``ABC`` or ``ABC.csv``
    fname = symbol if symbol.lower().endswith(".csv") else f"{symbol}.csv"
    data_path = os.path.join(os.getcwd(), "backend", "data", fname)
    if not os.path.exists(data_path):
        return {"error": "Stock data not found."}
    df = pd.read_csv(data_path)
    if "Close" not in df.columns:
        return {"error": "CSV must have 'Close' column."}
    closes = df["Close"].values
    # simple forecast: use last value and random walk
    last_close = float(closes[-1])
    returns = (closes[1:] - closes[:-1]) / closes[:-1]
    avg_return = float(np.mean(returns)) if len(returns) > 0 else 0
    predictions = []
    current = last_close
    for i in range(days):
        # apply avg return with small noise
        noise = np.random.normal(loc=0.0, scale=0.01)
        new_close = current * (1 + avg_return + noise)
        # generate OHLC with small variation
        high = max(current, new_close) * (1 + np.random.uniform(0.001, 0.01))
        low = min(current, new_close) * (1 - np.random.uniform(0.001, 0.01))
        volume = int(df['Volume'].iloc[-1] * (1 + np.random.normal(0, 0.1))) if 'Volume' in df.columns else 0
        open_price = current
        predictions.append({
            "Open": round(open_price, 2),
            "High": round(high, 2),
            "Low": round(low, 2),
            "Close": round(new_close, 2),
            "Volume": volume
        })
        current = new_close
    return predictions

# Example changes in backend_core.py (or backend/core.py)
def train_model(params):
    symbol = params.get("stock")
    time_window = params.get("time_window")
    epochs = params.get("epochs", 10)
    batch_size = params.get("batch_size", 32)
    if not symbol or time_window is None:
        return {"message": "Error: 'stock' and 'time_window' parameters required."}
    # Check CSV exists
    data_path = os.path.join(os.getcwd(), "backend", "data", f"{symbol}.csv")
    if not os.path.exists(data_path):
        return {"message": f"Stock data for {symbol} not found."}

    # Instantiate model with symbol, then train
    cnn_model = CNNLSTMModel(day=time_window, symbol=symbol)
    try:
        cnn_model.train(symbol, epochs=epochs, batch_size=batch_size)
    except Exception as e:
        return {"message": f"Error during training: {e}"}

    # Get status after training
    status = cnn_model.get_status()
    return {"message": f"Model trained for {symbol}.", "status": status}


def run_simulation(params):
    logging.info("Running simulation with parameters: %s", params)
    symbol = params.get("symbol")
    strategy = params.get("strategy")
    initial_capital = params.get("initial_capital", 10000)
    if not symbol or not strategy:
        return {"message": "Error: symbol and strategy are required."}
    data = load_stock_data(symbol)
    if not data:
        return {"message": f"No data for symbol {symbol}."}
    # Instantiate agent using MyCustomAgent
    agent = MyCustomAgent(trend=data, initial_money=initial_capital, agent_name=strategy)
    states_buy, states_sell, gains, pnl_pct = agent.buy()
    # Update global portfolio
    global portfolio_value, total_profit, win_rate
    total_profit += gains
    portfolio_value += gains
    if hasattr(agent.agent, "get_win_rate"):
        try:
            win_rate = agent.agent.get_win_rate()
        except:
            pass
    return {
        "final_portfolio": round(portfolio_value, 2),
        "total_profit": round(total_profit, 2),
        "win_rate": round(win_rate, 2),
        "trades_executed": len(states_buy) + len(states_sell),
        "gains": round(gains, 2),
        "pnl_pct": round(pnl_pct, 2),
        "strategy": strategy,
        "symbol": symbol
    }

def get_portfolio_stats():
    return {
        "portfolio_value": round(portfolio_value, 2),
        "total_profit": round(total_profit, 2),
        "win_rate": round(win_rate, 2)
    }

def get_strategy_performance():
    strategies = [
        {"name": "Evolution Agent", "profit": round(np.random.uniform(1000, 2000), 2), "win_rate": round(np.random.uniform(60, 80), 1)},
        {"name": "MA Crossover", "profit": round(np.random.uniform(1000, 2000), 2), "win_rate": round(np.random.uniform(60, 80), 1)},
        {"name": "RSI Strategy", "profit": round(np.random.uniform(1000, 2000), 2), "win_rate": round(np.random.uniform(60, 80), 1)},
        {"name": "CNN-LSTM", "profit": round(np.random.uniform(1000, 2000), 2), "win_rate": round(np.random.uniform(60, 80), 1)},
        {"name": "Signal Rolling", "profit": round(np.random.uniform(1000, 2000), 2), "win_rate": round(np.random.uniform(60, 80), 1)}
    ]
    strategies = sorted(strategies, key=lambda x: x['profit'], reverse=True)
    return strategies

def get_model_status():
    # Dummy status for demonstration
    return {
        "status": "Active",
        "accuracy": 0.0,
        "mae": 0.0,
        "last_trained": "N/A"
    }
