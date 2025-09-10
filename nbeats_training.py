import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# N-BEATS block definition (generic, as in the paper)
class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.fc5(x)

class NBeats(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, n_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_dim, hidden_dim, output_dim) for _ in range(n_blocks)
        ])

    def forward(self, x):
        y = 0
        for block in self.blocks:
            y = y + block(x)
        return y

def train_nbeats(X_train, y_train, X_val, y_val, input_dim, model_path, epochs=20, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NBeats(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze()
                val_loss = criterion(pred, yb)
                val_losses.append(val_loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {np.mean(val_losses):.6f}")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train N-BEATS models for specified forecast intervals.")
    parser.add_argument("symbol", type=str, help="Symbol to train on (e.g., XNO_USDT)")
    parser.add_argument("--window_sizes", type=int, nargs='+', default=[30, 60, 120], help="TDA window sizes to use")
    parser.add_argument("--forecast_intervals", type=int, nargs='+', default=[1], help="Forecast intervals (in steps, e.g., 1 for next, 5 for 5 steps ahead)")
    parser.add_argument("--features_path", type=str, default=None, help="Path to features CSV (default: processed_data/{symbol}_features.csv)")
    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)
    features_path = args.features_path or f'processed_data/{args.symbol}_features.csv'
    df = pd.read_csv(features_path)
    target_col = 'close'
    for ws in args.window_sizes:
        feature_cols = [col for col in df.columns if f'tda_ws{ws}_' in col]
        features = df[feature_cols].values
        for interval in args.forecast_intervals:
            # Target: close price shifted by -interval
            target = df[target_col].shift(-interval).values
            mask = ~np.isnan(features).any(axis=1) & ~np.isnan(target)
            X = features[mask]
            y = target[mask]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            model_path = f'models/{args.symbol}_ws{ws}_f{interval}_nbeats.pth'
            print(f"Training N-BEATS for {args.symbol}, window size {ws}, forecast interval {interval}...")
            train_nbeats(X_train, y_train, X_val, y_val, X_train.shape[1], model_path)
    print("All models trained and saved.")
