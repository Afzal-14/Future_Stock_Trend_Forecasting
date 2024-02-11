# Stock Trend Forecasting with LSTM

# Introduction
This repository contains a Python script for predicting future stock prices and forecasting trends using Long Short-Term Memory (LSTM) neural networks. The code utilizes historical stock price data obtained from Yahoo Finance and implements a sequence-to-sequence LSTM model for training and testing.

# Requirements
Ensure you have the necessary Python libraries installed. You can install them using the following:

pip install yfinance pandas numpy matplotlib scikit-learn tensorflow

# Usage
Run the script and enter the stock symbol when prompted.

stock_symbol = input("Enter Stock Symbol:")

Define the training and testing date ranges.

train_start_date = "2010-01-01"

train_end_date = "2022-07-31"

test_start_date = "2022-08-01"

test_end_date = date.today() - timedelta(1)

forecast_days = 10

Execute the script, and the model will download historical stock data, train an LSTM model, and generate predictions and future forecasts.

# Implementation Details
## Data Import and Processing
The script uses the yfinance library to download historical stock price data.

The data is processed, and the closing prices are normalized using Min-Max scaling.

## Model Training
The LSTM model is defined and trained on the training data.

The model architecture includes multiple LSTM layers with dropout for regularization.

The model is compiled using the Adam optimizer and mean squared error loss.

## Model Testing
The script downloads test data and preprocesses it for testing the trained model.

The model predicts the stock prices on the test data.

# Future Trend Forecasting
Future stock prices are forecasted using the trained model.

The script generates a plot showing the actual stock prices, predicted prices, and future forecasts.

# Visualization
The script produces two plots:

A plot showing actual stock prices, predicted prices, and future forecasts.

![uber](https://github.com/Afzal-14/Future_Stock_Trend_Forecasting/assets/120948536/19e77b4f-7c40-41f1-843a-49453113a1e3)


A separate plot focusing on the future stock price forecasts.

![Capture](https://github.com/Afzal-14/Future_Stock_Trend_Forecasting/assets/120948536/5042badc-e31d-4578-8927-64dc5f35dee5)


# Deployment
The model is deployed for web development through Streamlit. 

Deployment link --> Click here https://futurestocktrendforecasting.streamlit.app/

# Conclusion
This code provides a foundation for stock price prediction and trend forecasting using LSTM neural networks. Users can further customize and enhance the model for specific requirements. Remember that stock market predictions involve inherent risks, and this script should be used for educational and experimental purposes only.

