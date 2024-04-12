#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:24:45 2024

@author: tnguyen287
"""

import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def plot_stock_data(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Date'], data['Close'], marker='', color='blue', linewidth=2)
    
    plt.title('Stock Data Over One Year')
    plt.xlabel('Date')
    plt.ylabel('Close Price ($)')

    plt.show()


stock_ticker = input("Enter the stock ticker symbol (e.g., 'AAPL' for Apple Inc.): ")

# Fetch data using the user's input
df = yf.Ticker(stock_ticker)
df_history = df.history(start='2023-01-01')

# Reset index to make 'Date' a column
df_history.reset_index(inplace=True)

# Call the plotting function
plot_stock_data(df_history)

data = df_history['Close']
dataset = data.values.reshape(-1, 1) # Reshape to 2D array for scaling
# Get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * 0.8) #80% of the data set

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Create the training dataset
#create the scaled training data set
train_data = scaled_data[0:training_data_len, :]
#split the data into x_train and y_train data sets
x_train = []
y_train = []
for i in range (60, len (train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i <=60:
        print(x_train)
        print(y_train)
        print()

#convert the x_train and y_train to numpy arrays
knn_model = KNeighborsRegressor(n_neighbors=5)
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
knn_model.fit(x_train, y_train)
x_test = np.array(x_test)

# Fit the model with training data
predictions = knn_model.predict(x_test)
# Make predictions on the test data
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_unscaled = scaler.inverse_transform(np.array(y_test[60:]).reshape(-1, 1))

# Calculate RMSE using the test data and the predictions
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'The RMSE of the KNN predictions is: {rmse}')

train = df_history[:training_data_len].copy()
valid = df_history[training_data_len:].copy()

# Add 'Predictions' to 'valid'
valid['Predictions'] = predictions

# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price ($)', fontsize=18)
plt.plot(train['Date'], train['Close'], label='Train')
plt.plot(valid['Date'], valid['Close'], label='Actual', alpha=0.7)
plt.plot(valid['Date'], valid['Predictions'], label='Prediction', alpha=0.7)
plt.legend(loc='upper left', fontsize=12)
plt.show()

