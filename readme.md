# Flask API for Machine Learning Models

This project provides a Flask-based REST API to interact with machine learning models for stock price prediction. The API integrates functionality to fetch stock data, train models (Linear Regression and LSTM), and combine predictions.

---

## Features

- **Fetch Stock Data**: Retrieve historical stock data using Yahoo Finance.
- **Train Models**:
  - Linear Regression
  - LSTM (Long Short-Term Memory)
- **Combine Predictions**: Merge predictions from both models.
- **Retrieve Combined Predictions**: Access the combined results in JSON format.

---


## Requirements

- Python 3.7+
- Flask
- pandas
- numpy
- scikit-learn
- tensorflow
- yfinance
- joblib
- matplotlib

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd project_root
Install dependencies:


    pip install -r requirements.txt

Run the Flask API:



    python flask_ml_api.py
Access the API at:


    http://127.0.0.1:5000/
---

## API Endpoints

### 1. Fetch Stock Data
**GET** `/fetch-stock-data`

- **Query Parameters**:
  - `ticker` (str): Stock ticker symbol (default: `AAPL`)
  - `start_date` (str): Start date (format: `YYYY-MM-DD`, default: `2010-01-01`)
  - `end_date` (str): End date (format: `YYYY-MM-DD`, optional)

- **Response**:
  - JSON object containing stock data.

### 2. Train Linear Regression Model
**POST** `/train-lr`

- **Body**:
  - Upload a CSV file with stock data.

- **Response**:
  - Training status message.

### 3. Train LSTM Model
**POST** `/train-lstm`

- **Body**:
  - Upload a CSV file with stock data.

- **Response**:
  - Training status message.

### 4. Combine Predictions
**GET** `/combine-predictions`

- **Response**:
  - Combination status message.

### 5. Retrieve Combined Predictions
**GET** `/predictions`

- **Response**:
  - JSON object containing combined predictions.

---

## Example Usage

 ### Fetch Stock Data
    ```bash
    curl "http://127.0.0.1:5000/fetch-stock-data?ticker=AAPL&start_date=2020-01-01&end_date=2021-01-01"

# Acknowledgments
### Yahoo
### Finance API
### Flask
### scikit-learn
### TensorFlow
    


