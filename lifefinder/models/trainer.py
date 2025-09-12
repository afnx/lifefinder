import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from typing import Optional


class Trainer:
    """
    Trainer class to handle training and evaluation of the ExoplanetNN model.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3, device: Optional[str] = None):
        """
        Args:
            model (nn.Module): The model to train.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            device (Optional[str], optional): Device to train on. Defaults to None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0
        for X, y in train_loader:
            # Move data to device
            X, y = X.to(self.device), y.to(self.device).float().unsqueeze(1)

            # Forward pass
            preds = self.model(X)
            loss = self.criterion(preds, y)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)

    def evaluate(self, val_loader):
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(self.device)
                y = y.to(self.device).float().unsqueeze(1)
                preds = self.model(X).cpu().numpy().ravel()
                y_true.extend(y.cpu().numpy().ravel())
                y_pred.extend((preds > 0.5).astype(int))
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)
        }
