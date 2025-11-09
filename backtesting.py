import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from nbeats_model import NBeats

def backtest_strategy(df, model, feature_cols, scaler):
    """
    Runs the backtest strategy on a given dataframe using a trained model.
    """
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().detach().numpy().squeeze()

    # Simple strategy: buy if next price predicted > current price, else sell
    # Ensure preds and close values are aligned and have no NaNs
    valid_indices = ~np.isnan(preds) & ~np.isnan(df['close'].values)
    preds = preds[valid_indices]
    close_prices = df['close'].values[valid_indices]

    if len(preds) < 2 or len(close_prices) < 2:
        return np.array([1.0]) # Not enough data to backtest

    signals = np.where(preds > close_prices, 1, -1)
    
    # Align signals and returns
    returns = np.diff(close_prices) / close_prices[:-1]
    signals = signals[:-1] # Signals predict the *next* step, so align with returns of that step
    
    strat_returns = signals * returns
    equity_curve = np.cumprod(1 + strat_returns)
    return equity_curve

def backtest_short_only_strategy(df, model, feature_cols, scaler):
    """
    Runs a short-only backtest strategy on a given dataframe using a trained model.
    """
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    with torch.no_grad():
        preds = model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().detach().numpy().squeeze()

    # Short-only strategy: sell if next price predicted < current price, else hold cash (0)
    valid_indices = ~np.isnan(preds) & ~np.isnan(df['close'].values)
    preds = preds[valid_indices]
    close_prices = df['close'].values[valid_indices]

    if len(preds) < 2 or len(close_prices) < 2:
        return np.array([1.0])

    signals = np.where(preds < close_prices, -1, 0) # -1 for short, 0 for hold
    
    returns = np.diff(close_prices) / close_prices[:-1]
    signals = signals[:-1]
    
    strat_returns = signals * returns
    equity_curve = np.cumprod(1 + strat_returns)
    return equity_curve

def buy_and_hold(df):
    """
    Calculates the equity curve for a simple buy-and-hold strategy.
    """
    if len(df['close']) < 2:
        return np.array([1.0])
    returns = np.diff(df['close'].values) / df['close'].values[:-1]
    equity_curve = np.cumprod(1 + returns)
    return equity_curve

def moving_average_crossover(df, short=10, long=50):
    """
    Calculates the equity curve for a moving average crossover strategy.
    """
    if len(df['close']) < long:
        return np.array([1.0])
    short_ma = df['close'].rolling(short).mean()
    long_ma = df['close'].rolling(long).mean()
    
    signals = np.where(short_ma > long_ma, 1, -1)
    returns = np.diff(df['close'].values) / df['close'].values[:-1]
    
    # Align signals and returns, accounting for MA lookback period
    signals = signals[long-1:-1]
    returns = returns[long-1:]

    if len(signals) != len(returns):
        min_len = min(len(signals), len(returns))
        signals = signals[:min_len]
        returns = returns[:min_len]

    strat_returns = signals * returns
    equity_curve = np.cumprod(1 + strat_returns)
    return equity_curve

def plot_results(results, title, json_output=None):
    """
    Plots the equity curves from backtesting results.
    """
    plt.figure(figsize=(12, 7))
    for label, curve in results.items():
        if len(curve) > 1:
            plt.plot(curve, label=label)
    plt.title(title)
    plt.xlabel('Time Steps (Test Period)')
    plt.ylabel('Equity (Normalized)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if not json_output:
        plt.show()
    else:
        # Save final equity results as JSON
        import json
        final_equity = {label: curve[-1] for label, curve in results.items()}
        with open(json_output, 'w') as json_file:
            json.dump(final_equity, json_file)
        print(f"[INFO] Final equity results saved to {json_output}")

if __name__ == "__main__":
    import subprocess
    import argparse
    import random
    from datetime import datetime, timedelta

    parser = argparse.ArgumentParser(description="Train and backtest N-BEATS models on fresh random data.")
    parser.add_argument("symbol", type=str, help="Symbol to backtest (e.g., XNO_USDT)")
    parser.add_argument("--window_sizes", type=int, nargs='+', default=[30], help="TDA window sizes to use")
    parser.add_argument("--forecast_intervals", type=int, nargs='+', default=[1], help="Forecast intervals to use")
    parser.add_argument("--tick", type=str, default="1m", help="Timeframe/tick for data acquisition (default: 1m)")
    parser.add_argument("--timehorizon", type=int, default=1000, help="How many units of tick to fetch (default: 1000)")
    parser.add_argument("--startingdate", type=str, default=None, help="ISO8601 start date for data acquisition")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data to use for testing (default: 0.2)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for on-the-fly training")
    parser.add_argument("--json_output", type=str, default=None, help="Path to save final equity results as JSON instead of plotting.")
    args = parser.parse_args()

    import sys
    venv_python = sys.executable

    # 1. Generate a random starting date if not provided
    if not args.startingdate:
        now = datetime.utcnow()
        two_years_ago = now - timedelta(days=2*365)
        random_days = random.randint(0, (now - two_years_ago).days)
        random_date = two_years_ago + timedelta(days=random_days)
        args.startingdate = random_date.strftime('%Y-%m-%dT%H:%M:%S')
        print(f"[INFO] Random starting date for backtest: {args.startingdate}")

    # 2. Pull new data
    print(f"[INFO] Pulling new data for {args.symbol}...")
    symbol_ccxt = args.symbol.replace('_', '/')
    data_dir = 'temp_data'
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, f'{args.symbol}_{args.tick}.csv')
    cmd = [
        venv_python, "data_acquisition.py", symbol_ccxt,
        "--tick", args.tick, "--timehorizon", str(args.timehorizon), "--startingdate", args.startingdate
    ]
    subprocess.run(cmd, check=True)
    orig_data_path = f'data/{args.symbol}_{args.tick}.csv'
    if os.path.exists(orig_data_path):
        os.replace(orig_data_path, data_path)

    # 3. Run TDA feature engineering
    print(f"[INFO] Running TDA feature engineering for {args.symbol}...")
    features_path = os.path.join(data_dir, f'{args.symbol}_features.csv')
    cmd = [venv_python, "tda_feature_engineering.py", args.symbol, "--input_path", data_path, "--output_path", features_path]
    for ws in args.window_sizes:
        cmd += ["--window_sizes", str(ws)]
    subprocess.run(cmd, check=True)

    # 4. Train and backtest for each combination
    df_features = pd.read_csv(features_path)
    results = {}
    target_col = 'close'
    
    baselines_calculated = False

    for ws in args.window_sizes:
        for f in args.forecast_intervals:
            print(f"\n[PROCESS] Running for window size {ws}, forecast interval {f}")
            feature_cols = [col for col in df_features.columns if f'tda_ws{ws}_' in col]
            if not feature_cols:
                print(f"Skipping ws{ws} (no TDA features in current data)")
                continue

            # Prepare data
            target = df_features[target_col].shift(-f).values
            mask = ~np.isnan(df_features[feature_cols]).any(axis=1) & ~np.isnan(target)
            X = df_features[feature_cols][mask].values
            y = target[mask]
            
            if len(X) < 100:
                print(f"Skipping ws{ws}/f{f} (not enough valid data: {len(X)} points)")
                continue

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, shuffle=False)
            df_test = df_features[mask].iloc[len(X_train):].reset_index(drop=True)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Train model
            print(f"Training N-BEATS for ws{ws}, f{f}...")
            model = NBeats(input_dim=X_train.shape[1])
            train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            for epoch in range(args.epochs):
                model.train()
                epoch_loss = 0
                for i, (xb, yb) in enumerate(train_loader):
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    pred = model(xb).squeeze()
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{args.epochs}, Avg Loss: {epoch_loss/len(train_loader):.6f}")
            
            model.eval()
            
            # Backtest on the test set
            label_long_short = f'N-BEATS L/S ws{ws} f{f}'
            results[label_long_short] = backtest_strategy(df_test, model, feature_cols, scaler)

            label_short_only = f'N-BEATS Short-Only ws{ws} f{f}'
            results[label_short_only] = backtest_short_only_strategy(df_test, model, feature_cols, scaler)

            # Add baselines once, using the first valid test set
            if not baselines_calculated and not df_test.empty:
                print("[INFO] Calculating baselines on the current test set...")
                results['Buy & Hold'] = buy_and_hold(df_test)
                results['MA Crossover'] = moving_average_crossover(df_test)
                baselines_calculated = True

    if not results:
         print("\n[ERROR] No models were successfully trained or backtested. Not enough data or features.")
    else:
        plot_results(results, f'{args.symbol} - On-the-Fly Training & Backtest', args.json_output)
