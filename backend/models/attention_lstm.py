import os
import time
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Attention, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import Accuracy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import log  # but use np.log for series

class AttentionLSTMModel:
    def __init__(self, day=3, symbol=None):
        """
        Initialize model with window size `day` and stock `symbol`.
        """
        self.day = day
        self.symbol = symbol
        # Build the model architecture based on the window size and 6 features
        self.model = self.build_model()
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.history = None
        self.last_trained = None

    def build_model(self):
        """
        Build an Attention Multilayer LSTM model. Input shape is (timesteps=day, features=6).
        """
        model = Sequential()
        # Input layer with shape (day, 6)
        model.add(InputLayer(input_shape=(self.day, 6)))
        # Multilayer LSTM (2x)
        model.add(LSTM(units=64, activation="swish", return_sequences=True))
        model.add(LSTM(units=64, activation="swish", return_sequences=True))
        # Attention adaptation
        model.add(Attention())
        # Output for binary classification
        model.add(Dense(units=1, activation="sigmoid"))

        # Compile with binary crossentropy loss
        optim = Adam(learning_rate=0.01, decay=1e-5)
        model.compile(loss="binary_crossentropy", optimizer=optim, metrics=["accuracy"])
        return model

    def load_data(self, symbol=None):
        """
        Load CSV data for `symbol`, compute normalized features, and create sliding windows.
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
        D = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Define ret and zscore
        ret = lambda x, y: np.log(y / x)
        zscore = lambda x: (x - x.mean()) / x.std()

        # Compute features
        Res = pd.DataFrame()
        Res['c_2_o'] = zscore(ret(D['Open'], D['Close']))
        Res['h_2_o'] = zscore(ret(D['Open'], D['High']))
        Res['l_2_o'] = zscore(ret(D['Open'], D['Low']))
        Res['c_2_h'] = zscore(ret(D['High'], D['Close']))
        Res['h_2_l'] = zscore(ret(D['Low'], D['High']))
        Res['vol'] = zscore(D['Volume'])

        # Compute label basis
        c1_c0 = ret(D['Close'], D['Close'].shift(-1)).fillna(0)

        # Build sliding windows
        features = Res
        num_samples = len(features)
        X, y = [], []
        for i in range(self.day - 1, num_samples - 1):
            window_start = i - self.day + 1
            window = features.iloc[window_start:i + 1].values  # shape (day, 6)
            label = 1 if c1_c0.iloc[i] > 0 else 0
            X.append(window)
            y.append(label)
        X = np.array(X)  # shape (N, day, 6)
        y = np.array(y)  # shape (N,)

        # Split into training and test sets (80% train, 20% test)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

    def train(self, symbol, epochs=100, batch_size=64):
        """
        Train the model for the given symbol, epochs, and batch size.
        """
        logging.info("Training Attention-LSTM for symbol=%s, window=%d, epochs=%d, batch=%d",
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
        filepath = MODEL_NAME + "-{epoch:02d}-{val_accuracy:.3f}.h5"
        checkpoint = ModelCheckpoint(
            os.path.join(save_path, filepath),
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            mode="max"
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
        loss, acc = self.model.evaluate(self.test_x, self.test_y)
        return {
            "status": "Active" if self.last_trained else "Training Required",
            "accuracy": round(acc * 100, 1),
            "mae": round(loss, 3),  # using loss as proxy
            "last_trained": self.last_trained or "N/A",
        }

    def save_model(self, filepath=None):
        if filepath is None:
            filename = f"AttentionLSTM_{self.symbol}_day{self.day}_{int(time.time())}.h5"
            filepath = os.path.join(os.getcwd(), filename)
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)