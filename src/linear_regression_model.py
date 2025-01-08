import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os
import matplotlib.pyplot as plt

def train_linear_regression_model(data, ticker_symbol):
    """
    Train and evaluate a linear regression model on the given stock data.

    Args:
        data (pd.DataFrame): The stock data to train the model on.
        ticker_symbol (str): The stock ticker symbol for saving predictions.
    """
    # Select features and target variable
    features = ['Close', 'High', 'Low', 'Open', 'Volume', 'EPS', 'Revenue', 'ROE', 'P/E']
    X = data[features]
    y = data['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Save the model
    model_path = f'models/{ticker_symbol}_regression_model.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Predict future stock prices (example with last row repeated)
    future_data = pd.DataFrame([X.iloc[-1]] * 3)  # Example of future predictions
    future_predictions = model.predict(future_data)

    # Save future predictions
    future_data['Predicted Close'] = future_predictions
    future_data.to_csv(f'data/{ticker_symbol}_lr_predictions.csv', index=False)
    print(f"Linear Regression predictions saved to 'data/{ticker_symbol}_lr_predictions.csv'")

