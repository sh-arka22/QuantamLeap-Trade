# File: backend/api.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from backend.core import (
    list_csv_files,
    upload_csv,
    get_predictions,
    train_model,
    run_simulation,
    get_portfolio_stats,
    get_strategy_performance,
    get_model_status
)
import os

app = Flask(__name__)

# Endpoint to list available CSVs
@app.route('/csv_files', methods=['GET'])
def csv_files():
    files = list_csv_files()
    return jsonify(files)

# Endpoint to upload new CSV
@app.route('/upload_csv', methods=['POST'])
def upload_csv_endpoint():
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400
    filename = secure_filename(file.filename)
    result = upload_csv(file)
    return jsonify(result)

# Endpoint to train model on CSV data
@app.route('/train_model', methods=['POST'])
def train_model_endpoint():
    params = request.json
    result = train_model(params)
    return jsonify(result)

# Endpoint to predict 30-day OHLCV
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symbol = data.get("stock")
    days = data.get("days", 30)
    predictions = get_predictions(symbol, days=days)
    return jsonify(predictions)

# Endpoint to run trading simulation
@app.route('/run_simulation', methods=['POST'])
def run_simulation_endpoint():
    params = request.json
    app.logger.info("Running simulation with parameters:/api.py %s", params)
    result = run_simulation(params)
    return jsonify(result)

# Endpoint to get portfolio stats
@app.route('/portfolio', methods=['GET'])
def portfolio_endpoint():
    stats = get_portfolio_stats()
    return jsonify(stats)

# Endpoint for strategy performance breakdown
@app.route('/strategy_performance', methods=['GET'])
def strategy_performance():
    stats = get_strategy_performance()
    return jsonify(stats)

# Endpoint for model status/metrics
@app.route('/model_status', methods=['GET'])
def model_status():
    status = get_model_status()
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True)
