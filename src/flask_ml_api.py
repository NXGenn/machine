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
    return "Welcome to the Flask API! Use the endpoints to predict stock prices."

@app.route('/fetch-stock-data', methods=['GET'])
def fetch_stock():
    ticker_symbol = request.args.get('ticker', 'AAPL')
    start_date = request.args.get('start_date', '2010-01-01')
    end_date = request.args.get('end_date', None)

    try:
        data = fetch_stock_data(ticker_symbol, start_date, end_date)
        if data is not None:
            return jsonify({"message": f"Stock data for {ticker_symbol} fetched successfully.", 
                            "data": data.to_dict(orient='records')}), 200
        else:
            return jsonify({"message": "No data found for the specified stock"}), 404
    except Exception as e:
        return jsonify({"message": "Error fetching stock data", "error": str(e)}), 500

@app.route('/train-lr', methods=['POST'])
def train_lr():
    try:
        ticker_symbol = request.form.get('ticker', 'AAPL')
        data_path = f"data/stock_data.csv"
        if not os.path.exists(data_path):
            return jsonify({"message": f"No data found for {ticker_symbol}. Fetch data first."}), 400
        
        data = pd.read_csv(data_path)
        train_linear_regression_model(data)  # Pass ticker_symbol
        return jsonify({"message": f"Linear Regression model trained successfully for {ticker_symbol}."}), 200
    except Exception as e:
        return jsonify({"message": "Error training Linear Regression model", "error": str(e)}), 500

@app.route('/train-lstm', methods=['POST'])
def train_lstm():
    try:
        ticker_symbol = request.form.get('ticker')  # Fetch the 'ticker' parameter from form data
        if not ticker_symbol:
            return jsonify({"message": "Ticker symbol is missing in the request."}), 400

        data_path = f"data/stock_data.csv"
        if not os.path.exists(data_path):
            return jsonify({"message": f"No data found . Fetch data first."}), 400

        # Load the data and train the model
        data = pd.read_csv(data_path)
        train_lstm_model(data)
        return jsonify({"message": f"LSTM model trained successfully ."}), 200
    except Exception as e:
        return jsonify({"message": "Error training LSTM model", "error": str(e)}), 500

@app.route('/combine-predictions', methods=['GET'])
def combine():
    try:
        ticker_symbol = request.args.get('ticker', 'AAPL')  # Default to 'AAPL' if no ticker is provided
        combine_predictions()
        return jsonify({"message": f"Predictions combined successfully ."}), 200
    except Exception as e:
        return jsonify({"message": f"Error combining predictions .", "error": str(e)}), 500


@app.route('/predictions', methods=['GET'])
def get_predictions():
    try:
        ticker_symbol = request.args.get('ticker', 'AAPL')
        combined_path = f"data/combined_predictions.csv"
        if os.path.exists(combined_path):
            data = pd.read_csv(combined_path)
            return jsonify({"message": f"Combined predictions  fetched successfully.", 
                            "data": data.to_dict(orient='records')}), 200
        else:
            return jsonify({"message": f"Combined predictions not found ."}), 404
    except Exception as e:
        return jsonify({"message": "Error fetching combined predictions", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")

