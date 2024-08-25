import robin_stocks.robinhood as r
import yfinance as yf
import talib
import numpy as np
import pandas as pd
import schedule
import time
from datetime import datetime, timedelta

r.login('your_email', 'your_password')

# fetch stock tickers from an ETF
def get_etf_holdings(etf_symbol):
    etf = yf.Ticker(etf_symbol)
    holdings = etf.history(period="1d")['Close'].index
    components = etf.get_holdings()['Symbol'].tolist()
    return components

# filter stocks based on price range and analyst rating
def filter_stocks(stocks, min_price=10, max_price=100):
    filtered_stocks = []
    for stock in stocks:
        try:
            fundamentals = r.stocks.get_fundamentals(stock)[0]
            price = float(fundamentals['last_trade_price'])
            rating = r.stocks.get_ratings(stock)['summary']['rating']['buy']
            if min_price <= price <= max_price and rating > 0:
                filtered_stocks.append(stock)
        except Exception as e:
            print(f"Error fetching data for {stock}: {e}")
    return filtered_stocks

# fetch historical data from Robinhood
def fetch_ohlcv(symbol, interval='hour', span='week'):
    historicals = r.stocks.get_stock_historicals(symbol, interval=interval, span=span)
    df = pd.DataFrame(historicals)
    df['timestamp'] = pd.to_datetime(df['begins_at'])
    df['open'] = df['open_price'].astype(float)
    df['high'] = df['high_price'].astype(float)
    df['low'] = df['low_price'].astype(float)
    df['close'] = df['close_price'].astype(float)
    df['volume'] = df['volume'].astype(int)
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

# calculate EMA
def calculate_ema(df, period):
    return talib.EMA(df['close'].values, timeperiod=period)

# check if 21 EMA is within x% of 55 EMA
def is_ema_close(df, ema_short=21, ema_long=55, threshold=0.5):
    ema21 = calculate_ema(df, ema_short)
    ema55 = calculate_ema(df, ema_long)
    difference = abs(ema21[-1] - ema55[-1])
    avg_price = (ema21[-1] + ema55[-1]) / 2
    percentage_difference = (difference / avg_price) * 100
    return percentage_difference <= threshold

# place a market order in Robinhood
def place_order(symbol, amount, order_type='buy'):
    if order_type == 'buy':
        order = r.orders.order_buy_option_limit('open', 'buy_to_open', amount, symbol, 1.00)  # Adjust price as needed
    else:
        order = r.orders.order_sell_option_limit('close', 'sell_to_close', amount, symbol, 1.00)  # Adjust price as needed
    return order

# check daily RSI and sell if it exceeds 60
def check_rsi_and_sell(symbol, option_id, amount, rsi_threshold=60):
    df_daily = fetch_ohlcv(symbol, interval='day', span='month')
    rsi_daily = talib.RSI(df_daily['close'].values, timeperiod=14)
    
    if rsi_daily[-1] > rsi_threshold:
        order = place_order(option_id, amount, order_type='sell')
        print(f"Sell Order placed: {order}")
        return True
    return False

# monitor and enforce stop-loss
def monitor_stop_loss(symbol, option_id, amount, purchase_price, stop_loss_threshold=0.70):
    while True:
        current_price = r.options.get_option_market_data_by_id(option_id)[0]['adjusted_mark_price']
        if float(current_price) <= purchase_price * stop_loss_threshold:
            order = place_order(option_id, amount, order_type='sell')
            print(f"Stop-loss triggered. Sell Order placed: {order}")
            return True
    return False

# top 10 stocks with close EMA crossover
def trading_bot_for_top_stocks(etf_symbol, threshold=0.5, strike_offset=1, ema_short=21, ema_long=55, expiration_offset=30, rsi_threshold=60, stop_loss_threshold=0.70):
    # Get ETF components
    components = get_etf_holdings(etf_symbol)
    
    # Filter stocks by price and analyst rating
    filtered_stocks = filter_stocks(components)
    
    for symbol in filtered_stocks:
        df = fetch_ohlcv(symbol)
        
        # Check if 21 EMA is within x% of 55 EMA
        if is_ema_close(df, ema_short, ema_long, threshold):
            print(f"{symbol}: EMA21 is within {threshold}% of EMA55")

            expiration_date = (datetime.utcnow() + timedelta(days=expiration_offset)).strftime('%Y-%m-%d')
            options = r.options.find_options_by_expiration_and_strike(symbol, expiration_date)
            otm_call = options[strike_offset]  # Adjust based on desired OTM position
            
            amount = 1
            
            # buy order
            buy_order = place_order(otm_call['id'], amount)
            purchase_price = float(buy_order['price'])
            print(f"Buy Order placed for {symbol}: {buy_order}")

            # monitor stop-loss and RSI to sell
            if not monitor_stop_loss(symbol, otm_call['id'], amount, purchase_price, stop_loss_threshold):
                check_rsi_and_sell(symbol, otm_call['id'], amount, rsi_threshold)
        else:
            print(f"{symbol}: No close EMA crossover detected.")

# bot scheduled to run once per hour during trading hours
def run_bot():
    trading_days = pd.bdate_range(start=datetime.now().date(), end=datetime.now().date()).tolist()
    if datetime.now().date() in trading_days:
        print(f"Running bot at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        etf_symbol = 'XLF'  # Example ETF symbol for Financials
        trading_bot_for_top_stocks(etf_symbol, threshold=0.5)  # Modify threshold as needed

# run bot every hour
schedule.every().hour.at(":00").do(run_bot)

#
while True:
    schedule.run_pending()
    time.sleep(1)
