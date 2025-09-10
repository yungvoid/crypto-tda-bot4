import ccxt
import pandas as pd
import os
from datetime import datetime, timedelta

def fetch_data(symbol, timeframe='1m', since=None, limit=1000):
    """Fetches historical data for a given symbol."""
    exchange = ccxt.binance()
    if since is None:
        # Default to fetching data for the last 30 days
        since = exchange.parse8601((datetime.utcnow() - timedelta(days=30)).isoformat())
    
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        if len(ohlcv) == 0:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 1

    return all_ohlcv

def save_to_csv(symbol, data):
    """Saves data to a CSV file."""
    if not os.path.exists('data'):
        os.makedirs('data')
    
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    filename = f"data/{symbol.replace('/', '_')}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved data for {symbol} to {filename}")

if __name__ == "__main__":
    cryptos = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
    for crypto in cryptos:
        print(f"Fetching data for {crypto}...")
        data = fetch_data(crypto)
        save_to_csv(crypto, data)
