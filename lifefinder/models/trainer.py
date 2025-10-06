import torch
import torch.nn as nn
import torch.optim as optim
import json

from sklearn.metrics import accuracy_score, f1_score
from typing import Optional


class Trainer:
    """
    Trainer class to handle training and evaluation of the ExoplanetNN model.
    """

    def __init__(
        self, model: nn.Module, lr: float = 1e-3, device: Optional[str] = None
    ):
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

    def train_epoch(self, train_dataloader):
        self.model.train()
        epoch_loss = 0
        for X, y in train_dataloader:
            # Move data to device
            X, y = X.to(self.device), y.to(self.device).float()

            # Forward pass
            outputs = self.model(X).squeeze()
            loss = self.criterion(outputs, y)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
        return epoch_loss / len(train_dataloader)

    def evaluate(self, val_dataloader, threshold: float = 0.5):
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for X, y in val_dataloader:
                X = X.to(self.device)
                y = y.to(self.device).float().unsqueeze(1)
                outputs = self.model(X).squeeze()

                # Convert outputs to binary predictions
                preds = (outputs > threshold).int()

                y_true.extend(y.numpy())
                y_pred.extend(preds)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return {"accuracy": acc, "f1": f1}

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    @staticmethod
    def log_metrics(metrics: dict, path):
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
