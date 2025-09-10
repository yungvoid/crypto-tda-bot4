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
    cryptos = ['BTC_USDT', 'ETH_USDT', 'ADA_USDT']
    window_sizes = [30, 60, 120]
    target_col = 'close'
    os.makedirs('models', exist_ok=True)

    for crypto in cryptos:
        df = pd.read_csv(f'processed_data/{crypto}_features.csv')
        for ws in window_sizes:
            # Select TDA features for this window size
            feature_cols = [col for col in df.columns if f'tda_ws{ws}_' in col]
            features = df[feature_cols].values
            # Target: next-step close price (shifted by -1)
            target = df[target_col].shift(-1).values
            # Remove rows with NaN (from padding or last row)
            mask = ~np.isnan(features).any(axis=1) & ~np.isnan(target)
            X = features[mask]
            y = target[mask]
            # Train/test split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            # Train model
            model_path = f'models/{crypto}_ws{ws}_nbeats.pth'
            print(f"Training N-BEATS for {crypto}, window size {ws}...")
            train_nbeats(X_train, y_train, X_val, y_val, X_train.shape[1], model_path)
    print("All models trained and saved.")
