
import ccxt
import pandas as pd
import os
from datetime import datetime, timedelta
import sys

def fetch_data(symbol, timeframe='1m', since=None, limit=1000, total=None):
    """
    Fetch historical OHLCV data for a given symbol and timeframe from Binance.
    Returns a list of OHLCV records or an empty list on error.
    Shows a terminal loading bar for progress.
    """
    exchange = ccxt.binance()
    supported_timeframes = exchange.timeframes.keys()
    if timeframe not in supported_timeframes:
        print(f"[ERROR] The requested tick/timeframe '{timeframe}' is not supported by Binance. Supported: {list(supported_timeframes)}")
        return []
    if since is None:
        since = exchange.parse8601((datetime.utcnow() - timedelta(days=30)).isoformat())
    all_ohlcv = []
    fetched = 0
    bar_length = 40
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        except Exception as e:
            print(f"[ERROR] Exception while fetching data: {e}")
            break
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        fetched += len(ohlcv)
        # Loading bar
        if total:
            progress = min(fetched / total, 1.0)
            block = int(round(bar_length * progress))
            bar = '#' * block + '-' * (bar_length - block)
            sys.stdout.write(f"\r[Fetching] |{bar}| {fetched}/{total} records")
            sys.stdout.flush()
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < limit or (total and fetched >= total):
            break
    if total:
        sys.stdout.write('\n')
    return all_ohlcv

def save_to_csv(symbol, data, tick=None):
    """
    Save OHLCV data to a CSV file, including tick in filename if provided.
    """
    if not os.path.exists('data'):
        os.makedirs('data')
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    filename = f"data/{symbol.replace('/', '_')}_{tick}.csv" if tick else f"data/{symbol.replace('/', '_')}.csv"
    df.to_csv(filename, index=False)
    print(f"[INFO] Saved data for {symbol} to {filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch historical OHLCV data for a symbol from Binance.")
    parser.add_argument("symbol", type=str, help="Symbol to fetch (e.g., BTC/USDT)")
    parser.add_argument("--tick", type=str, default="1m", help="Timeframe/tick (e.g., 1m, 5m, 1h, 1d)")
    parser.add_argument("--timehorizon", type=int, default=None, help="How many units of tick to fetch (default depends on tick)")
    parser.add_argument("--startingdate", type=str, default=None, help="ISO8601 start date (e.g., 2023-01-01T00:00:00)")
    args = parser.parse_args()

    # Small sensible defaults for timehorizon based on tick
    default_horizons = {"1m": 300, "5m": 100, "15m": 50, "1h": 24, "1d": 7}
    tick = args.tick
    timehorizon = args.timehorizon if args.timehorizon is not None else default_horizons.get(tick, 50)

    since = None
    if args.startingdate:
        try:
            since = ccxt.binance().parse8601(args.startingdate)
        except Exception as e:
            print(f"[ERROR] Could not parse startingdate: {e}")
            exit(1)

    print(f"[INFO] Fetching data for {args.symbol} with tick {tick}, timehorizon {timehorizon}, startingdate {args.startingdate if args.startingdate else 'default'}...")
    data = []
    fetched = 0
    limit = 1000
    while fetched < timehorizon:
        batch = fetch_data(args.symbol, timeframe=tick, since=since, limit=min(limit, timehorizon-fetched), total=timehorizon)
        if not batch:
            break
        data.extend(batch)
        fetched += len(batch)
        since = batch[-1][0] + 1
        if len(batch) < limit:
            break
    if data:
        save_to_csv(args.symbol, data, tick=tick)
    else:
        print(f"[WARN] No data fetched for {args.symbol} with tick {tick}.")
