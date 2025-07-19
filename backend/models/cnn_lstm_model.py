# File: backend/models/cnn_lstm_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, LSTM, Conv2D, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import time
import logging

class CNNLSTMModel:
    def __init__(self, day=3, symbol=None):
        """
        Initialize model with window size `day` and stock `symbol`.
        """
        self.day = day
        self.symbol = symbol
        # Build the model architecture based on the window size and 5 features
        self.model = self.build_model()
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.history = None
        self.last_trained = None

    def build_model(self):
        """
        Build a CNN-LSTM model. Input shape is (timesteps=day, width=1, channels=5).
        """
        model = Sequential()
        # Input layer with shape (day, 1, 5)
        model.add(InputLayer(input_shape=(self.day, 1, 5)))
        # Conv2D over each timestep (1x1 conv with 5 input channels)
        model.add(
            Conv2D(
                filters=32,
                kernel_size=(1, 1),
                activation="swish",
                padding="same"
            )
        )
        # Reshape output (batch, day, 1, 32) -> (batch, day, 32) for LSTM
        model.add(Reshape((self.day, 32)))
        model.add(LSTM(units=64, activation="swish"))
        model.add(Dense(units=1, activation="swish"))  # single-value output (next-day Close)

        # Compile with MAE loss
        optim = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-5)
        model.compile(loss="mae", optimizer=optim, metrics=["mae"])
        return model

    def load_data(self, symbol=None):
        """
        Load CSV data for `symbol`, normalize OHLCV, and create sliding windows.
        """
        # Determine symbol (parameter overrides instance attribute if provided)
        symbol = symbol or self.symbol
        if symbol is None:
            raise ValueError("Stock symbol must be provided for loading data.")
        # Load CSV from backend/data
        data_dir = os.path.join(os.getcwd(), "backend", "data")
        csv_path = os.path.join(data_dir, f"{symbol}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file for {symbol} not found.")
        # Read CSV and select OHLCV columns
        df = pd.read_csv(csv_path)
        # Drop 'Date' and 'Adj Close' if present
        if 'Date' in df.columns:
            df = df.drop(columns=['Date'])
        if 'Adj Close' in df.columns:
            df = df.drop(columns=['Adj Close'])
        # Ensure we only have OHLCV columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Convert to numpy and scale features to [0,1] range
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(df.values)  # shape (num_samples, 5)
        num_samples = scaled_values.shape[0]

        # Build sliding windows of length self.day
        X, y = [], []
        for i in range(num_samples - self.day):
            window = scaled_values[i:i + self.day]      # shape (day, 5)
            next_close = scaled_values[i + self.day][3] # index 3 is 'Close'
            X.append(window)
            y.append(next_close)
        X = np.array(X)  # shape (N, day, 5)
        y = np.array(y).reshape(-1, 1)  # shape (N, 1)

        # Split into training and test sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        # Reshape for CNN-LSTM: (N, day, 1, 5)
        self.train_x = X_train.reshape(-1, self.day, 1, 5)
        self.test_x = X_test.reshape(-1, self.day, 1, 5)
        self.train_y = y_train
        self.test_y = y_test

    def train(self, symbol, epochs=100, batch_size=64):
        """
        Train the model for the given symbol, epochs, and batch size.
        """
        logging.info("Training CNN-LSTM for symbol=%s, window=%d, epochs=%d, batch=%d",
                     symbol, self.day, epochs, batch_size)
        # Load data (symbol passed)
        self.load_data(symbol)
        if self.train_x is None or self.train_y is None:
            print("No training data available; skipping training.")
            return
        # Prepare directories and checkpoint callback
        MODEL_NAME = f"NEXT{self.day}-{symbol}-{int(time.time())}"
        SUBDIR = f"day{self.day}"
        save_path = os.path.join(os.getcwd(), "backend", "models", SUBDIR)
        os.makedirs(save_path, exist_ok=True)
        filepath = MODEL_NAME + "-{epoch:02d}-{val_mae:.3f}.h5"
        checkpoint = ModelCheckpoint(
            os.path.join(save_path, filepath),
            monitor="val_mae",
            verbose=1,
            save_best_only=True,
            mode="min"
        )
        # Fit the model
        self.history = self.model.fit(
            self.train_x, self.train_y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.test_x, self.test_y),
            callbacks=[checkpoint]
        )
        self.last_trained = time.strftime("%Y-%m-%d %H:%M:%S")

    def predict(self, X):
        return self.model.predict(X)

    def get_status(self):
        if self.history is None:
            return {"status": "Inactive", "accuracy": 0.0, "mae": 0.0, "last_trained": "Never"}
        if self.test_x is None or self.test_y is None:
            return {"status": "Trained (no test data)", "accuracy": 0.0, "mae": 0.0, "last_trained": self.last_trained}
        val_pred = self.model.predict(self.test_x)
        mae = mean_absolute_error(self.test_y, val_pred)
        rmse = np.sqrt(mean_squared_error(self.test_y, val_pred))
        r2 = r2_score(self.test_y, val_pred) * 100
        return {
            "status": "Active" if self.last_trained else "Training Required",
            "accuracy": round(r2, 1),
            "mae": round(mae, 3),
            "last_trained": self.last_trained or "N/A",
        }

    def save_model(self, filepath=None):
        if filepath is None:
            filename = f"CNNLSTM_{self.symbol}_day{self.day}_{int(time.time())}.h5"
            filepath = os.path.join(os.getcwd(), filename)
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
