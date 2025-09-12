import torch
import torch.nn as nn
import torch.nn.functional as F


class ExoplanetNN(nn.Module):
    """
    A feedforward neural network for habitability classification.
    Outputs a probability of being habitable (binary classification).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim // 2, 1)  # Binary output

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.out(x))  # Probability [0,1]
        return x
