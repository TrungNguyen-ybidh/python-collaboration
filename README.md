import csv

def read_stock_data(file_name):
    data = []
    with open(file_name, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def simple_moving_average(data, window_size):
    sma_values = []
    for i in range(len(data)):
        if i + 1 < window_size:
            sma_values.append(None)  # Not enough data points to calculate SMA
        else:
            sma = sum(float(data[j]['Close']) for j in range(i-window_size+1, i+1)) / window_size
            sma_values.append(sma)
    return sma_values

import matplotlib.pyplot as plt

def plot_stock_data(dates, prices, sma=None):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, prices, label='Stock Price')
    
    if sma is not None:
        plt.plot(dates, sma, label='SMA', color='orange')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Stock Price Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
def linear_regression(x, y):
    N = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
    sum_xx = sum(x_i**2 for x_i in x)
    
    # Calculate the slope (m)
    m = ((N * sum_xy) - (sum_x * sum_y)) / ((N * sum_xx) - (sum_x**2))
    
    # Calculate the y-intercept (b)
    b = (sum_y - (m * sum_x)) / N
    
    return m, b
