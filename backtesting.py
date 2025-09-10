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
    cryptos = ['BTC_USDT', 'ETH_USDT']
    window_sizes = [30, 60, 120]
    for crypto in cryptos:
        df = pd.read_csv(f'processed_data/{crypto}_features.csv')
        for ws in window_sizes:
            model_path = f'models/{crypto}_ws{ws}_nbeats.pth'
            feature_cols = [col for col in df.columns if f'tda_ws{ws}_' in col]
            # Remove rows with NaN
            mask = ~np.isnan(df[feature_cols]).any(axis=1)
            df_valid = df[mask].reset_index(drop=True)
            if not os.path.exists(model_path) or len(df_valid) < 100:
                print(f"Skipping {crypto} ws{ws} (model or data missing)")
                continue
            print(f"Backtesting {crypto} window size {ws}...")
            # Prepare scaler as in training
            scaler = StandardScaler()
            scaler.fit(df_valid[feature_cols].values)
            # Load model
            model = load_model(model_path, len(feature_cols))
            # Backtest strategies
            results = {
                'N-BEATS TDA': backtest_strategy(df_valid, model, feature_cols, scaler),
                'Buy & Hold': buy_and_hold(df_valid),
                'MA Crossover': moving_average_crossover(df_valid)
            }
            plot_results(results, f'{crypto} (window {ws})')
