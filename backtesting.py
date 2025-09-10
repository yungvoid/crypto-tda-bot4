import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class NBeatsBlock(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc5(x)

class NBeats(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, n_blocks=3):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            NBeatsBlock(input_dim, hidden_dim, output_dim) for _ in range(n_blocks)
        ])
    def forward(self, x):
        y = 0
        for block in self.blocks:
            y = y + block(x)
        return y

def load_model(model_path, input_dim):
    model = NBeats(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def backtest_strategy(df, model, feature_cols, scaler):
    X = df[feature_cols].values
    X = scaler.transform(X)
    preds = model(torch.tensor(X, dtype=torch.float32)).detach().numpy().squeeze()
    # Simple strategy: buy if next price predicted > current price, else sell
    signals = np.where(preds > df['close'].values, 1, -1)
    returns = np.diff(df['close'].values) / df['close'].values[:-1]
    # Align signals and returns
    signals = signals[:-1]
    strat_returns = signals * returns
    equity_curve = np.cumprod(1 + strat_returns)
    return equity_curve

def buy_and_hold(df):
    returns = np.diff(df['close'].values) / df['close'].values[:-1]
    equity_curve = np.cumprod(1 + returns)
    return equity_curve

def moving_average_crossover(df, short=10, long=50):
    short_ma = df['close'].rolling(short).mean()
    long_ma = df['close'].rolling(long).mean()
    signals = np.where(short_ma > long_ma, 1, -1)
    returns = np.diff(df['close'].values) / df['close'].values[:-1]
    signals = signals[:-1]
    strat_returns = signals * returns
    equity_curve = np.cumprod(1 + strat_returns)
    return equity_curve

def plot_results(results, title):
    plt.figure(figsize=(10,6))
    for label, curve in results.items():
        plt.plot(curve, label=label)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Equity (normalized)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import subprocess
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Backtest and compare multiple N-BEATS models and baselines.")
    parser.add_argument("symbol", type=str, help="Symbol to backtest (e.g., XNO_USDT)")
    parser.add_argument("--window_sizes", type=int, nargs='+', default=[30, 60, 120], help="TDA window sizes to use")
    parser.add_argument("--forecast_intervals", type=int, nargs='+', default=[1, 3, 15], help="Forecast intervals to use")
    parser.add_argument("--models", type=str, nargs='*', default=None, help="Explicit list of model paths to backtest")
    parser.add_argument("--tick", type=str, default="1m", help="Timeframe/tick for data acquisition (default: 1m)")
    parser.add_argument("--timehorizon", type=int, default=None, help="How many units of tick to fetch (default depends on tick)")
    parser.add_argument("--startingdate", type=str, default=None, help="ISO8601 start date for data acquisition")
    args = parser.parse_args()


    import sys
    venv_python = sys.executable
    import random
    from datetime import datetime, timedelta

    # 1. Generate a random starting date in the last two years if not provided
    if not args.startingdate:
        now = datetime.utcnow()
        two_years_ago = now - timedelta(days=2*365)
        random_days = random.randint(0, (now - two_years_ago).days)
        random_date = two_years_ago + timedelta(days=random_days)
        args.startingdate = random_date.strftime('%Y-%m-%dT%H:%M:%S')
        print(f"[INFO] Random starting date for backtest: {args.startingdate}")

    # 2. Pull new data for the symbol and save in backtest_data
    print(f"[INFO] Pulling new data for {args.symbol}...")
    symbol_ccxt = args.symbol.replace('_', '/')
    backtest_data_dir = 'backtest_data'
    os.makedirs(backtest_data_dir, exist_ok=True)
    backtest_data_path = os.path.join(backtest_data_dir, f'{args.symbol}_{args.tick}.csv')
    cmd = [
        venv_python, "data_acquisition.py", symbol_ccxt,
        "--tick", args.tick,
        "--timehorizon", str(args.timehorizon or 100),
        "--startingdate", args.startingdate
    ]
    subprocess.run(cmd, check=True)
    orig_data_path = f'data/{args.symbol}_{args.tick}.csv'
    if os.path.exists(orig_data_path):
        os.replace(orig_data_path, backtest_data_path)

    # 3. Run TDA feature engineering for the backtest data and save in backtest_features
    print(f"[INFO] Running TDA feature engineering for {args.symbol} (backtest data)...")
    backtest_features_dir = 'backtest_features'
    os.makedirs(backtest_features_dir, exist_ok=True)
    backtest_features_path = os.path.join(backtest_features_dir, f'{args.symbol}_features.csv')
    cmd = [venv_python, "tda_feature_engineering.py", args.symbol, "--input_path", backtest_data_path, "--output_path", backtest_features_path]
    for ws in args.window_sizes:
        cmd += ["--window_sizes", str(ws)]
    subprocess.run(cmd, check=True)

    # 4. Proceed with backtesting as before, using backtest_features
    df = pd.read_csv(backtest_features_path)
    results = {}

    # Discover models if not explicitly provided
    model_paths = args.models
    if model_paths is None:
        model_paths = []
        for ws in args.window_sizes:
            for f in args.forecast_intervals:
                path = f'models/{args.symbol}_ws{ws}_f{f}_nbeats.pth'
                if os.path.exists(path):
                    model_paths.append(path)

    for model_path in model_paths:
        # Parse window size and forecast interval from filename
        try:
            ws = int(model_path.split('_ws')[1].split('_')[0])
            f = int(model_path.split('_f')[1].split('_')[0])
        except Exception:
            print(f"Could not parse window size/forecast interval from {model_path}, skipping.")
            continue
        feature_cols = [col for col in df.columns if f'tda_ws{ws}_' in col]
        if not feature_cols:
            print(f"Skipping {model_path} (no TDA features for ws{ws} in current data)")
            continue
        mask = ~np.isnan(df[feature_cols]).any(axis=1)
        df_valid = df[mask].reset_index(drop=True)
        if len(df_valid) < 100:
            print(f"Skipping {model_path} (not enough valid data)")
            continue
        scaler = StandardScaler()
        scaler.fit(df_valid[feature_cols].values)
        model = load_model(model_path, len(feature_cols))
        label = f'N-BEATS ws{ws} f{f}'
        results[label] = backtest_strategy(df_valid, model, feature_cols, scaler)

    # Add baselines (on all valid data for the smallest window size)
    min_ws = min(args.window_sizes)
    feature_cols = [col for col in df.columns if f'tda_ws{min_ws}_' in col]
    if feature_cols:
        mask = ~np.isnan(df[feature_cols]).any(axis=1)
        df_valid = df[mask].reset_index(drop=True)
        results['Buy & Hold'] = buy_and_hold(df_valid)
        results['MA Crossover'] = moving_average_crossover(df_valid)
    else:
        print(f"[WARN] No TDA features for window size {min_ws}, skipping baselines.")

    plot_results(results, f'{args.symbol} - Model & Baseline Comparison')
