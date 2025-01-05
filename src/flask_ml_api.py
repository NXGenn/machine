from flask import Flask, request, jsonify
from fetch_data import fetch_stock_data
from linear_regression_model import train_linear_regression_model
from lstm_model import train_lstm_model
from main import combine_predictions
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Flask API! Use the defined endpoints to interact with the application."

@app.route('/favicon.ico')
def favicon():
    return '', 204

# (Existing endpoints below...)
@app.route('/fetch-stock-data', methods=['GET'])
def fetch_stock():
    ticker_symbol = request.args.get('ticker', 'AAPL')
    start_date = request.args.get('start_date', '2010-01-01')
    end_date = request.args.get('end_date', None)

    try:
        data = fetch_stock_data(ticker_symbol, start_date, end_date)
        if data is not None:
            return jsonify({"message": "Stock data fetched successfully", "data": data.to_dict(orient='records')}), 200
        else:
            return jsonify({"message": "No data found"}), 404
    except Exception as e:
        return jsonify({"message": "Error fetching stock data", "error": str(e)}), 500

# Endpoint to train the Linear Regression model
@app.route('/train-lr', methods=['POST'])
def train_lr():
    try:
        csv_file = request.files['file']
        data = pd.read_csv(csv_file)
        train_linear_regression_model(data)
        return jsonify({"message": "Linear Regression model trained successfully"}), 200
    except Exception as e:
        return jsonify({"message": "Error training Linear Regression model", "error": str(e)}), 500

# Endpoint to train the LSTM model
@app.route('/train-lstm', methods=['POST'])
def train_lstm():
    try:
        csv_file = request.files['file']
        data = pd.read_csv(csv_file)
        train_lstm_model(data)
        return jsonify({"message": "LSTM model trained successfully"}), 200
    except Exception as e:
        return jsonify({"message": "Error training LSTM model", "error": str(e)}), 500

# Endpoint to combine predictions
@app.route('/combine-predictions', methods=['GET'])
def combine():
    try:
        combine_predictions()
        return jsonify({"message": "Predictions combined successfully"}), 200
    except Exception as e:
        return jsonify({"message": "Error combining predictions", "error": str(e)}), 500

# Endpoint to fetch combined predictions
@app.route('/predictions', methods=['GET'])
def get_predictions():
    try:
        combined_path = 'data/combined_predictions.csv'
        if os.path.exists(combined_path):
            data = pd.read_csv(combined_path)
            return jsonify({"message": "Combined predictions fetched successfully", "data": data.to_dict(orient='records')}), 200
        else:
            return jsonify({"message": "Combined predictions not found"}), 404
    except Exception as e:
        return jsonify({"message": "Error fetching combined predictions", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
