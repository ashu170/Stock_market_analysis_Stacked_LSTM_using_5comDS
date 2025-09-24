
import pandas as pd
df=pd.read_csv('combined_stock_data.csv')
df.head()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf

# Plotting styles
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# Define stock list and time range
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA']
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# Download data
company_data = {}
for stock in tech_list:
    company_data[stock] = yf.download(stock, start=start, end=end)

# Add company names and concatenate data
company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON", "TESLA"]
for stock, name in zip(tech_list, company_name):
    company_data[stock]["company_name"] = name

df = pd.concat(company_data.values(), axis=0)
df.describe()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(15, 12))
plt.subplots_adjust(top=0.95, bottom=0.05)

for i, company in enumerate(company_list, 1):
    ax = plt.subplot(3, 2, i)
    company['Close'].plot(ax=ax, label=tech_list[i - 1], color='tab:blue')
    ax.set_ylabel('Close')
    ax.set_title(f"Closing Price of {tech_list[i - 1]}")
    ax.legend()

    # Improve date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Hide the last (6th) subplot if there are only 5 companies
if len(company_list) < 6:
    plt.subplot(3, 2, 6).axis('off')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.figure(figsize=(15, 12))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

for i, company in enumerate(company_list, 1):
    ax = plt.subplot(3, 2, i)
    company['Volume'].plot(ax=ax, label=tech_list[i - 1], color='tab:blue')
    ax.set_ylabel('Volume')
    ax.set_title(f"Sales Volume for {tech_list[i - 1]}")
    ax.legend(loc='upper right')

    # Format date on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# If less than 6 stocks, hide the last empty subplot
if len(company_list) < 6:
    plt.subplot(3, 2, 6).axis('off')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

ma_day = [10, 20, 50]

# Compute moving averages
for ma in ma_day:
    for company in company_list:
        column_name = f"MA for {ma} days"
        company[column_name] = company['Close'].rolling(ma).mean()

# Set company names in the same order as company_list
company_names = ['APPLE', 'GOOGLE', 'MICROSOFT', 'AMAZON', 'TESLA']

# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
axes = axes.flatten()

# Plot for each company
for i, (df, name) in enumerate(zip(company_list, company_names)):
    df[['Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(ax=axes[i])
    axes[i].set_title(name)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Price')

# Hide the unused 6th subplot
if len(company_list) < len(axes):
    axes[-1].axis('off')

plt.tight_layout()
plt.show()

## What was the daily return of the stock on average
import matplotlib.pyplot as plt

# Calculate daily returns using 'Close' instead of 'Adj Close'
for company in company_list:
    company['Daily Return'] = company['Close'].pct_change()

# Names and plot layout
company_names = ['APPLE', 'GOOGLE', 'MICROSOFT', 'AMAZON', 'TESLA']

# Create a 3x2 grid of plots (enough for 5 companies)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
axes = axes.flatten()

# Plot each company's daily return
for i, (df, name) in enumerate(zip(company_list, company_names)):
    df['Daily Return'].plot(ax=axes[i], legend=True, linestyle='--', marker='o')
    axes[i].set_title(name)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Daily Return')

# Turn off the unused 6th subplot
if len(company_list) < len(axes):
    axes[-1].axis('off')

plt.tight_layout()
plt.show()

import yfinance as yf
import pandas as pd

# List of tech stocks
tech_list = ['AAPL', 'GOOGL', 'AMZN', 'MSFT','TSLA']

# Define start and end dates
start = '2020-01-01'
end = '2025-01-01'

# Fetch the stock data
data = yf.download(tech_list, start=start, end=end)

# Print the columns to check the available ones
print("Available columns:", data.columns)

# If 'Adj Close' exists, use it
if 'Adj Close' in data.columns:
    closing_prices = data['Adj Close']
else:
    # If 'Adj Close' is missing, fall back to 'Close' or any other available column
    closing_prices = data['Close']  # Use 'Close' as fallback

# Display the adjusted close (or fallback close) prices
print(closing_prices.head())

import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# List of tech stocks
tech_list = ['AAPL', 'GOOGL', 'AMZN', 'MSFT','TSLA']

# Define start and end dates
start = '2020-01-01'
end = '2025-01-01'

# Fetch the stock data
data = yf.download(tech_list, start=start, end=end)

# Print the columns to check the available ones
print("Available columns:", data.columns)

# Check if 'Adj Close' exists, else use 'Close'
if 'Adj Close' in data.columns:
    closing_prices = data['Adj Close']
else:
    closing_prices = data['Close']

# Calculate daily returns (percentage change)
tech_rets = closing_prices.pct_change()

# Now you can create the joint plot
sns.jointplot(x='GOOGL', y='GOOGL', data=tech_rets, kind='scatter', color='seagreen')

# Show the plot
plt.show()

# We can simply call pairplot on our DataFrame for an automatic visual analysis
# of all the comparisons

sns.pairplot(tech_rets, kind='reg')

# Set up our figure by naming it returns_fig, call PairPLot on the DataFrame
return_fig = sns.PairGrid(tech_rets.dropna())

# Using map_upper we can specify what the upper triangle will look like.
return_fig.map_upper(plt.scatter, color='purple')

# We can also define the lower triangle in the figure, inclufing the plot type (kde)
# or the color map (BluePurple)
return_fig.map_lower(sns.kdeplot, cmap='cool_d')

# Finally we'll define the diagonal as a series of histogram plots of the daily return
return_fig.map_diag(plt.hist, bins=30)

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.heatmap(tech_rets.corr(), annot=True, cmap='summer')
plt.title('Correlation of stock return')

## How much value do we put at risk by investing in a particular stock
rets = tech_rets.dropna()

area = np.pi * 20

plt.figure(figsize=(10, 8))
plt.scatter(rets.mean(), rets.std(), s=area)
plt.xlabel('Expected return')
plt.ylabel('Risk')

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(label, xy=(x, y), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom',
                 arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))

##Predicting the closing price stock price of APPLE inc
import yfinance as yf
import datetime

# Fetch the stock data
df = yf.download('AAPL', start='2016-01-01', end=datetime.datetime.now())

# Show the data
print(df)

plt.figure(figsize=(16,6))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

# Create a new dataframe with only the 'Close column
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))

training_data_len

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load data from yfinance
ticker = "AMZN"
df = yf.download(ticker, start="2015-01-01", end="2024-01-01")

# Extract just the 'Close' prices
dataset = df[['Close']].values  # Shape will be (n, 1)

# Initialize and scale
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Show result
print(scaled_data.shape)
print(scaled_data[:5])

import numpy as np
import yfinance as yf
import datetime
from sklearn.preprocessing import MinMaxScaler


# Initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale the data
scaled_data = scaler.fit_transform(dataset)

# Define train_data for training
train_data = scaled_data  # Use scaled data for training

# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])  # Last 60 days of data as input
    y_train.append(train_data[i, 0])      # The next day's closing price as output

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data to be 3D for LSTM input (samples, timesteps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Check the shape of x_train
print(x_train.shape)

from keras.models import Sequential
from keras.layers import Dense, LSTM

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
rmse

# Use AMZN Close prices
close_column = ('Close', 'AMZN')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Step 1: Download data
data = yf.download('AMZN', start='2012-01-01', end='2022-01-01')
data = data[['Close']]

# Step 2: Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Step 3: Create training dataset
training_data_len = int(np.ceil(len(scaled_data) * 0.8))

train_data = scaled_data[0:training_data_len, :]

# Create x_train and y_train
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape x_train
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Step 4: Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Step 5: Create testing dataset
test_data = scaled_data[training_data_len - 60:, :]
x_val = []
y_val = data['Close'].values[training_data_len:]

for i in range(60, len(test_data)):
    x_val.append(test_data[i-60:i, 0])

x_val = np.array(x_val)
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

# Step 6: Make predictions
predictions = model.predict(x_val)
predictions = scaler.inverse_transform(predictions)

# Step 7: Visualize
valid = data[training_data_len:].copy()
valid['Predictions'] = predictions

plt.figure(figsize=(16,6))
plt.title('LSTM Model: AMZN Stock Price Prediction')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(data['Close'][:training_data_len], label='Train')
plt.plot(valid['Close'], label='Validation')
plt.plot(valid['Predictions'], label='Predictions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load historical data
def load_data(ticker, start="2015-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end)
    return df['Close'].values.reshape(-1, 1), df

# Preprocess data and return train, val sets
def preprocess_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i-time_step:i])
        y.append(data_scaled[i])

    X, y = np.array(X), np.array(y)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_val, y_train, y_val, scaler

# Model building and training with validation
def build_and_train_model(X_train, X_val, y_train, y_val):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    return model

# Gradio prediction function
def predict_stock(ticker):
    data, df = load_data(ticker)
    if len(data) < 100:
        return "Error: Not enough data."

    time_step = 60
    X_train, X_val, y_train, y_val, scaler = preprocess_data(data, time_step)

    model = build_and_train_model(X_train, X_val, y_train, y_val)

    # Predict using validation set
    predicted_val = model.predict(X_val)
    predicted_prices = scaler.inverse_transform(predicted_val)
    actual_prices = scaler.inverse_transform(y_val.reshape(-1,1))

    # Plot predictions vs actuals
    plt.figure(figsize=(10, 5))
    plt.plot(actual_prices, label='Actual Price')
    plt.plot(predicted_prices, label='Predicted Price')
    plt.title(f'{ticker} - Validation Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    plt.savefig("plot.png")
    return "plot.png"

# Gradio UI
interface = gr.Interface(
    fn=predict_stock,
    inputs=gr.Textbox(label="Enter Stock Ticker (e.g., AMZN, AAPL)"),
    outputs=gr.Image(type="filepath", label="Predictions vs Actual Plot"),
    title="Stock Price Predictor"
)

interface.launch(share=True)

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Load data
ticker = "AMZN"
data = yf.download(ticker, start="2015-01-01", end="2024-01-01")

# Extract Close prices
data_close = data[['Close']]  # This gives you a DataFrame with one column

# Create train/valid split
training_data_len = int(len(data_close) * 0.8)
train = data_close[:training_data_len]
valid = data_close[training_data_len:].copy()

# Simulate predictions (replace with your real predictions)
import numpy as np
# For demo, we'll just simulate it as a slight shift from actual
predictions = valid['Close'].values * (1 + np.random.normal(0, 0.01, size=len(valid)))

# Add predictions
valid['Predictions'] = predictions

# Plotting
plt.figure(figsize=(16, 6))
plt.title('Model Predictions vs Actual (AMZN)')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price USD ($)', fontsize=14)
plt.plot(train.index, train['Close'], label='Train')
plt.plot(valid.index, valid['Close'], label='Validation')
plt.plot(valid.index, valid['Predictions'], label='Predictions')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

