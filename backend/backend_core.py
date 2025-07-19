import os
import numpy as np
import pandas as pd
import random
import json
from datetime import datetime
import math
import logging

from backend.models.cnn_lstm_model import CNNLSTMModel
from backend.agents.evolution_strategy import EvolutionStrategyAgent
from backend.agents.abcd_strategy import ABCDStrategyAgent
from backend.agents.signal_rolling import SignalRollingAgent
from backend.agents.my_custom_agent import MyCustomAgent  # your custom
from backend.portfolio import Portfolio
from backend.utils.data_utils import load_stock_data

# Module‐level shared objects
cnn_model = None
portfolio = Portfolio()

def list_csv_files():
    data_dir = os.path.join(os.getcwd(), "backend", "data")
    symbols = []
    try:
        for f in os.listdir(data_dir):
            if f.lower().endswith(".csv"):
                symbol, _ = os.path.splitext(f)
                symbols.append(symbol)
    except Exception:
        pass
    return symbols

def upload_csv(file_storage):
    data_dir = os.path.join(os.getcwd(), "backend", "data")
    try:
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, file_storage.filename)
        file_storage.save(filepath)
        return {"message": f"File {file_storage.filename} uploaded successfully."}
    except Exception as e:
        return {"message": f"Failed to upload file: {str(e)}"}

def train_model(params):
    import logging
    logging.info("Training model with parameters: %s", params)
    symbol = params.get("stock")
    time_window = params.get("time_window")
    epochs = params.get("epochs", 10)
    batch_size = params.get("batch_size", 32)

    if not symbol:
        return {"message": "Error: 'stock' parameter required."}
    data_path = os.path.join(os.getcwd(), "backend", "data", f"{symbol}.csv")
    if not os.path.exists(data_path):
        return {"message": f"Stock data for {symbol} not found."}
    if time_window is None:
        return {"message": "Error: 'time_window' parameter required."}

    global cnn_model
    try:
        cnn_model = CNNLSTMModel(day=time_window)
        cnn_model.train(epochs=epochs, batch_size=batch_size)
        status = cnn_model.get_status()
    except Exception as e:
        return {"message": f"Training error: {e}"}

    return {"message": f"Model trained for {symbol}.", "status": status}

def get_predictions(symbol, days=30):
    global cnn_model
    if cnn_model is None:
        return {"error": "Model not trained."}
    data_path = os.path.join(os.getcwd(), "backend", "data", f"{symbol}.csv")
    if not os.path.exists(data_path):
        return {"error": "Stock data not found."}
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return {"error": f"Error reading data: {e}"}

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        return {"error": "CSV must contain Open, High, Low, Close, Volume columns."}
    if len(df) < 10:
        return {"error": "Not enough data to make predictions."}

    last_rows = df.tail(10).copy()
    arr = []
    for _, row in last_rows.iterrows():
        vals = [row['Open'], row['High'], row['Low'], row['Close'], row['Volume'], 0.0]
        arr.append(vals)
    input_seq = np.array(arr).reshape((1, 10, 1, 6))
    predictions = []
    current_close = input_seq[0, -1, 0, 3]

    for _ in range(days):
        try:
            pred = cnn_model.predict(input_seq)
        except Exception as e:
            return {"error": f"Prediction error: {e}"}
        next_close = float(pred[0][0]) if hasattr(pred, 'shape') else float(pred)
        open_price = current_close
        high = max(current_close, next_close) * (1 + random.uniform(0.001, 0.01))
        low = min(current_close, next_close) * (1 - random.uniform(0.001, 0.01))
        volume = int(last_rows['Volume'].iloc[-1] * (1 + random.normalvariate(0, 0.1)))

        predictions.append({
            "Open": round(open_price, 2),
            "High": round(high, 2),
            "Low": round(low, 2),
            "Close": round(next_close, 2),
            "Volume": volume
        })

        current_close = next_close
        new_row = np.array([open_price, high, low, next_close, volume, 0.0]).reshape((1, 1, 1, 6))
        input_seq = np.concatenate((input_seq[:, 1:, :, :], new_row), axis=1)
        last_rows = last_rows.append({
            'Open': open_price, 'High': high, 'Low': low, 'Close': next_close, 'Volume': volume
        }, ignore_index=True).iloc[1:]

    return predictions

def run_simulation(params):
    """
    Run trading simulation using MyCustomAgent on historical and predicted data.
    """
    logging.info("Running simulation with parameters/backend_core.py: %s", params)
    symbol = params.get("symbol")
    strategy = params.get("strategy")
    initial_capital = params.get("initial_capital", 10000)
    days = params.get("days", 30)

    if not symbol or not strategy:
        return {"message": "Error: 'symbol' and 'strategy' are required."}

    try:
        data = load_stock_data(symbol)
    except FileNotFoundError:
        return {"message": f"No data for symbol '{symbol}'."}
    except KeyError as e:
        return {"message": str(e)}
    except Exception as e:
        return {"message": f"Error loading data: {e}"}

    closes_hist = [float(p) for p in data]
    if not closes_hist:
        return {"message": "No historical data available for symbol."}

    def simulate_series(prices, initial_money, strategy_name):
        trades = []
        agent = MyCustomAgent(trend=prices, initial_money=initial_money, agent_name=strategy_name)
        try:
            states_buy, states_sell, gains, pnl_pct = agent.buy()
        except Exception as e:
            return {"error": f"Strategy execution error: {e}"}

        current_money = initial_money
        inventory = 0
        values = []
        buy_set = set(states_buy)
        sell_set = set(states_sell)

        for i, price in enumerate(prices):
            if i in buy_set and current_money >= price:
                current_money -= price
                inventory += 1
                trades.append({"type": "buy", "timestamp": i, "price": price})
            if i in sell_set and inventory > 0:
                current_money += price
                inventory -= 1
                trades.append({"type": "sell", "timestamp": i, "price": price})
            values.append(round(current_money + inventory * price, 2))

        final_balance = values[-1] if values else initial_money
        profit = round(final_balance - initial_money, 2)
        profit_pct = round((final_balance - initial_money) / initial_money * 100, 2)

        # Sharpe ratio
        sharpe = None
        if len(values) > 1:
            rets = [(values[j] / values[j-1] - 1) for j in range(1, len(values)) if values[j-1] != 0]
            if rets:
                mean_ret, std_ret = np.mean(rets), np.std(rets)
                if std_ret:
                    sharpe = round(mean_ret / std_ret * math.sqrt(252), 2)

        # Max drawdown
        peak = values[0] if values else initial_money
        max_dd = 0.0
        for v in values:
            peak = max(peak, v)
            dd = (peak - v) / peak if peak else 0
            max_dd = max(max_dd, dd)
        max_drawdown_pct = round(max_dd * 100, 2)

        # Win rate
        wins, count_trades = 0, 0
        sell_idx = 0
        for b in states_buy:
            while sell_idx < len(states_sell) and states_sell[sell_idx] <= b:
                sell_idx += 1
            if sell_idx < len(states_sell):
                if prices[states_sell[sell_idx]] > prices[b]:
                    wins += 1
                count_trades += 1
                sell_idx += 1
        win_rate_pct = round(wins / count_trades * 100, 2) if count_trades else None

        return {
            "final_balance": round(final_balance, 2),
            "profit": profit,
            "profit_pct": profit_pct,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown_pct,
            "win_rate_pct": win_rate_pct,
            "trades_executed": len(states_buy) + len(states_sell),
            "trades": trades,
            "portfolio_values": values
        }

    # Historical simulation
    hist_result = simulate_series(closes_hist, initial_capital, strategy)
    results = {"historical": hist_result}

    # Predicted simulation
    if cnn_model is None:
        results["predicted"] = {"error": "CNNLSTM model not trained."}
    else:
        preds = get_predictions(symbol, days=days)
        if isinstance(preds, dict) and preds.get("error"):
            results["predicted"] = {"error": preds["error"]}
        else:
            pred_closes = [float(item["Close"]) for item in preds]
            if not pred_closes:
                results["predicted"] = {"error": "No predictions returned."}
            else:
                results["predicted"] = simulate_series(pred_closes, initial_capital, strategy)

    # Optionally save to JSON
    try:
        safe_strategy = strategy.replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(os.getcwd(), "backend", "results")
        os.makedirs(results_dir, exist_ok=True)
        filename = f"{symbol}_{safe_strategy}_{timestamp}.json"
        with open(os.path.join(results_dir, filename), "w") as f:
            json.dump(results, f, indent=4)
    except Exception:
        pass

    return results

def get_portfolio_stats():
    global portfolio
    return {
        "balance": round(portfolio.get_balance(), 2),
        "profit": round(portfolio.get_profit(), 2),
        "trades": len(portfolio.get_trades())
    }

def get_strategy_performance():
    strategies = ["Evolution Agent", "MA Crossover", "ABCD Strategy", "CNN-LSTM", "Signal Rolling"]
    perf = []
    for name in strategies:
        perf.append({
            "name": name,
            "profit": round(random.uniform(1000, 2000), 2),
            "win_rate": round(random.uniform(60, 80), 1)
        })
    perf.sort(key=lambda x: x["profit"], reverse=True)
    return perf

def get_model_status():
    global cnn_model
    if cnn_model is None:
        return {"status": "Inactive", "accuracy": 0.0, "mae": 0.0, "last_trained": "Never"}
    try:
        return cnn_model.get_status()
    except Exception as e:
        return {"status": "Error", "error": str(e)}
