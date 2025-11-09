import torch

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
