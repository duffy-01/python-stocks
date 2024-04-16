import math
import pandas_datareader as web
import yfinance as yf
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

df = yf.download('AAPL', start= '2014-01-01', end= '2023-01-01')
data = df[['Close']]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=25, batch_size=32)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

extended_period = 100  # for example, predict the next 100 days

# Create additional time steps for input data
extended_data = np.concatenate((scaled_data, np.zeros((extended_period, 1))))  # Extend data with zeros
for i in range(len(data), len(data) + extended_period):
    X_new = extended_data[i - time_step:i].reshape(1, time_step, 1)
    predicted_value = model.predict(X_new)[0][0]  # Predict next value
    extended_data[i, 0] = predicted_value  # Update the extended data with the predicted value

# Generate future dates for the extended period
future_dates = pd.date_range(start=df.index[-1], periods=extended_period + 1)

# Reshape the extended data for prediction
X_extended, _ = create_dataset(extended_data, time_step)
X_extended = X_extended.reshape(X_extended.shape[0], X_extended.shape[1], 1)

# Predict future values
future_predictions = model.predict(X_extended)

# Invert predictions to original scale
future_predictions = scaler.inverse_transform(future_predictions)

# Visualize predictions
plt.plot(df.index[:len(train_predictions)], data.iloc[:len(train_predictions), 0], label='Actual Train Data')
plt.plot(df.index[len(train_predictions) + time_step: len(train_predictions) + time_step + len(test_predictions)], data.iloc[len(train_predictions) + time_step: len(train_predictions) + time_step + len(test_predictions), 0], label='Actual Test Data')
plt.plot(df.index[time_step: len(train_predictions) + time_step], train_predictions, label='Train Predictions')
plt.plot(df.index[len(train_predictions) + time_step: len(train_predictions) + time_step + len(test_predictions)], test_predictions, label='Test Predictions')
plt.plot(future_dates[1:], future_predictions, label='Future Predictions')  # Use future_dates for x-values
plt.legend()
plt.show()