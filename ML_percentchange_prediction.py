import robin_stocks.robinhood as r
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from collections import defaultdict
import matplotlib.pyplot as plt

r.login('your_email', 'your_password')


def fetch_ohlcv(symbol, interval='hour', span='month'):
    historicals = r.stocks.get_stock_historicals(symbol, interval=interval, span=span)
    df = pd.DataFrame(historicals)
    df['timestamp'] = pd.to_datetime(df['begins_at'])
    df['open'] = df['open_price'].astype(float)
    df['high'] = df['high_price'].astype(float)
    df['low'] = df['low_price'].astype(float)
    df['close'] = df['close_price'].astype(float)
    df['volume'] = df['volume'].astype(int)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# prep data for LSTM model
def prepare_lstm_data(df, time_step=60):
    data = df['close'].values
    data = data.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# train LSTM model
def build_and_train_lstm(X_train, y_train, epochs=10, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

# predict next closing price
def predict_next_price(model, scaler, df, time_step=60):
    last_data = df['close'].values[-time_step:]
    last_data_scaled = scaler.transform(last_data.reshape(-1, 1))
    X_test = np.reshape(last_data_scaled, (1, time_step, 1))
    
    predicted_price_scaled = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    return predicted_price[0, 0]

# calculate daily percent changes
def calculate_percent_changes(df):
    df['percent_change'] = df['close'].pct_change() * 100
    df.dropna(inplace=True)
    return df

# Markov Chain
def build_markov_chain(df, n_states=10):
    percent_changes = df['percent_change'].values
    bins = np.linspace(min(percent_changes), max(percent_changes), n_states)
    states = np.digitize(percent_changes, bins) - 1
    
    transition_matrix = np.zeros((n_states, n_states))
    for (current_state, next_state) in zip(states[:-1], states[1:]):
        transition_matrix[current_state, next_state] += 1
    
    # normalize 
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    return bins, transition_matrix


def predict_next_state(markov_chain, current_state, bins):
    next_state = np.argmax(markov_chain[current_state])
    return bins[next_state]

# LSTM and Markov Chain trading logic
def trading_decision(symbol, lstm_model, scaler, df, bins, markov_chain, time_step=60):
    predicted_price = predict_next_price(lstm_model, scaler, df, time_step)
    last_percent_change = df['percent_change'].values[-1]
    current_state = np.digitize(last_percent_change, bins) - 1
    predicted_change = predict_next_state(markov_chain, current_state, bins)
    
    print(f"Predicted next hour's closing price for {symbol}: ${predicted_price:.2f}")
    print(f"Predicted percent change for next day: {predicted_change:.2f}%")
    
    if predicted_price > df['close'].values[-1] and predicted_change > 0:
        print(f"Decision: Buy {symbol}")

    else:
        print(f"Decision: Hold {symbol}")

# example usage:
symbol = 'AAPL'
df = fetch_ohlcv(symbol, interval='hour', span='month')

# prepare data and train LSTM model
X, y, scaler = prepare_lstm_data(df)
lstm_model = build_and_train_lstm(X, y)

# calculate percent changes and build Markov Chain
df = calculate_percent_changes(df)
bins, markov_chain = build_markov_chain(df)

trading_decision(symbol, lstm_model, scaler, df, bins, markov_chain)
