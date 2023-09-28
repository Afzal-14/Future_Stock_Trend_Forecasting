import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from datetime import date, timedelta
from keras.models import load_model
import streamlit as st


st.title("Future Stock Trend Forecasting")


stock_symbol = st.text_input("Enter Stock Symbol")
train_start_date = "2010-01-01"
train_end_date = "2022-07-31"
test_start_date = "2022-08-01"
test_end_date = date.today() - timedelta(1)
today_date = date.today()
forecast_days = 10

if st.button("Forecast"):

    df_train = yf.download(stock_symbol, start=train_start_date, end=train_end_date)
    df_test = yf.download(stock_symbol, start=test_start_date, end=test_end_date)


    st.subheader(f"{stock_symbol} Recent Data")
    st.write(df_test)


    st.subheader(f"{stock_symbol} Recent Trend")
    fig = plt.figure(figsize = (12,6))
    plt.plot(df_test.Close)
    st.pyplot(fig)


    train_data = df_train["Close"].values.reshape(-1,1)

    sc = MinMaxScaler()
    train_data = sc.fit_transform(train_data)


    X_train = []
    y_train = []
    sequence = 60

    for i in range(sequence, len(train_data)):           
        X_train.append(train_data[i-sequence:i, 0])
        y_train.append(train_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)


    model = Sequential()

    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1],1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")


    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=2)


    test_data = df_test["Close"].values.reshape(-1,1)

    scaled_test_data = sc.transform(test_data)


    X_test = []

    for i in range(sequence, len(scaled_test_data)):           
        X_test.append(scaled_test_data[i-sequence:i, 0])

    X_test = np.array(X_test)


    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_predict = model.predict(X_test)

    predict_price = sc.inverse_transform(y_predict)
    y_test = test_data[60:]



    st.subheader(f"{stock_symbol} Stock Price Prediction")
    fig1 = plt.figure(figsize=(12,6))
    plt.plot(df_test.index[60:], y_test, color="red", label="Actual Stock Price")
    plt.plot(df_test.index[60:], predict_price, color="green", label="Predicted Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    st.pyplot(fig1)


    forecast_dates = pd.date_range(start=today_date, periods=forecast_days)
    X_future = scaled_test_data
    forecast_data = []
    for i in range(forecast_days):    
        req = []    
        req.append(X_future[-sequence:, 0])
        X_input = np.array(req)
        X_reshape = np.reshape(X_input, (X_input.shape[0], X_input.shape[1], 1))
        y_forecast = model.predict(X_reshape)
        X_future = np.vstack((X_future, y_forecast))
        forecast_data.append(np.array(y_forecast[0]))  

    forecast_data = sc.inverse_transform(forecast_data)

    st.subheader(f"{stock_symbol} Stock Price Prediction & Future Forecasting")
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(df_test.index[-150:], y_test[-150:], color="red", label="Actual Stock Price")
    plt.plot(df_test.index[-150:], predict_price[-150:], color="green", label="Predicted Stock Price")
    plt.plot(forecast_dates, forecast_data, color="blue", label="Future Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    st.pyplot(fig2)


    st.subheader(f"{stock_symbol} Future Stock Price Forecasting")
    fig3 = plt.figure(figsize=(12,6))
    plt.plot(forecast_dates, forecast_data, color="blue", label="Future Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    st.pyplot(fig3)