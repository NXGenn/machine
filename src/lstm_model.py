import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_lstm_model(data, ticker_symbol):
    """
    Train and evaluate an LSTM model on the given stock data.

    Args:
        data (pd.DataFrame): The stock data to train the model on.
        ticker_symbol (str): The stock ticker symbol for saving predictions.
    """
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Select features and target variable
    features = ['Close', 'High', 'Low', 'Open', 'Volume', 'EPS', 'Revenue', 'ROE', 'P/E']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[features])

    def create_dataset(dataset, time_step=60):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step)])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)

    # Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, len(features))))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

    # Make predictions
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Save the model
    model_path = f'models/{ticker_symbol}_lstm_model.keras'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    # Save future predictions
    future_data = pd.DataFrame([X_test[-1]] * 3)  # Example for next 3 predictions
    future_predictions = model.predict(future_data)
    future_data['Predicted Close'] = future_predictions
    future_data.to_csv(f'data/{ticker_symbol}_lstm_predictions.csv', index=False)
    print(f"LSTM predictions saved to 'data/{ticker_symbol}_lstm_predictions.csv'")
