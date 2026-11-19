# models/neural_network.py
"""
Rete neurale (PyTorch) che stima il rendimento futuro.
PyTorch è facoltativo: se non è installato il resto del bot funziona lo stesso
(prima l'intero package `models` andava in crash all'import).
"""
import copy
import logging
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - dipende dall'ambiente
    TORCH_AVAILABLE = False

from .xgboost_model import chronological_split

logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:
    class TradingNet(nn.Module):
        """Rete feed-forward per il trading."""

        def __init__(self, input_dim: int, hidden_dims: List[int] = (128, 64),
                     dropout_rate: float = 0.3, output_dim: int = 1):
            super().__init__()
            layers, prev = [], input_dim
            for h in hidden_dims:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout_rate)]
                prev = h
            # uscita lineare: il target (standardizzato) non è limitato a [-1, 1]
            layers.append(nn.Linear(prev, output_dim))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)


class NeuralTradingModel:
    """Modello di rete neurale."""

    name = 'neural'

    def __init__(self,
                 hidden_dims: List[int] = (128, 64),
                 dropout_rate: float = 0.3,
                 learning_rate: float = 1e-3,
                 batch_size: int = 256,
                 epochs: int = 100,
                 early_stopping_patience: int = 15,
                 random_state: int = 42,
                 feature_names: Optional[List[str]] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch non installato: usa model_type='xgboost' oppure installa torch")
        self.hidden_dims = list(hidden_dims)
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.feature_names = feature_names

        self.model = None
        self.input_dim = None
        self.scaler_mean = None
        self.scaler_std = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        torch.manual_seed(self.random_state)
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if X_val is None or y_val is None:
            X, X_val, y, y_val = chronological_split(X, y)

        self.input_dim = X.shape[1]
        self.model = TradingNet(self.input_dim, self.hidden_dims, self.dropout_rate)

        self._fit_scaler(X)
        self.y_mean, self.y_std = float(y.mean()), float(y.std() or 1.0)
        Xs, Xvs = self._transform_scaler(X), self._transform_scaler(X_val)
        ys, yvs = (y - self.y_mean) / self.y_std, (y_val - self.y_mean) / self.y_std

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(Xs), torch.FloatTensor(ys).reshape(-1, 1)),
            batch_size=self.batch_size, shuffle=True, drop_last=len(Xs) > self.batch_size)
        X_val_t = torch.FloatTensor(Xvs)
        y_val_t = torch.FloatTensor(yvs).reshape(-1, 1)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.MSELoss()

        best_val, best_state, patience, epoch = float('inf'), None, 0, 0
        for epoch in range(self.epochs):
            self.model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(self.model(bx), by)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_loss = criterion(self.model(X_val_t), y_val_t).item()
            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val, patience = val_loss, 0
                # deepcopy: state_dict().copy() copiava solo i riferimenti ai tensori,
                # quindi il "miglior" modello veniva sovrascritto dagli epoch successivi
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.is_trained = True
        pred_val = self.predict(X_val)
        return {
            'best_val_loss': float(best_val),
            'epochs_trained': epoch + 1,
            'val_corr': float(np.corrcoef(pred_val, y_val)[0, 1]) if np.std(pred_val) > 0 else 0.0,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Modello non addestrato")
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.FloatTensor(self._transform_scaler(np.asarray(X, dtype=float))))
        return out.numpy().flatten() * self.y_std + self.y_mean

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pred = self.predict(X)
        prob = 1 / (1 + np.exp(-pred / (self.y_std or 1.0)))
        return np.vstack([1 - prob, prob]).T

    def get_feature_importance(self) -> Dict[str, float]:
        """Importanza stimata come gradiente medio assoluto sull'input (dati standardizzati)."""
        if not self.is_trained:
            return {}
        self.model.eval()
        X_test = torch.randn(1000, self.input_dim, requires_grad=True)
        grads = torch.autograd.grad(self.model(X_test).sum(), X_test)[0]
        importance = grads.abs().mean(dim=0).numpy()
        names = self.feature_names or [f'feature_{i}' for i in range(len(importance))]
        return {n: float(v) for n, v in zip(names, importance)}

    def _fit_scaler(self, X: np.ndarray):
        self.scaler_mean = X.mean(axis=0)
        self.scaler_std = X.std(axis=0)
        self.scaler_std[self.scaler_std < 1e-8] = 1

    def _transform_scaler(self, X: np.ndarray) -> np.ndarray:
        return (X - self.scaler_mean) / self.scaler_std
