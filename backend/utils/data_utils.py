# File: backend/utils/data_utils.py
import numpy as np
import pandas as pd
import os
import logging

def get_state(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]
    res = [block[i + 1] - block[i] for i in range(n - 1)]
    return np.array([res])

def load_stock_data(stock_symbol):
    """
    Load 'Close' prices from backend/data/{symbol}.csv.
    `stock_symbol` may be provided with or without '.csv'.
    """
    # Ensure we have just the filename
    fname = stock_symbol if stock_symbol.lower().endswith(".csv") else f"{stock_symbol}.csv"
    file_path = os.path.join(os.getcwd(), "backend", "data", fname)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    if 'Close' not in df.columns:
        raise KeyError(f"'Close' column missing in {fname}")

    return df['Close'].astype(float).tolist()


def calculate_performance(predictions, actual):
    mae = np.mean(np.abs(predictions - actual))
    rmse = np.sqrt(np.mean((predictions - actual) ** 2))
    r2 = 1 - (np.sum((actual - predictions) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
    return mae, rmse, r2
