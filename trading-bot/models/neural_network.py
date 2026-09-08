# models/neural_network.py
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)

class TradingNet(nn.Module):
    """Rete neurale per il trading."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.3,
                 output_dim: int = 1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class NeuralTradingModel:
    """Modello di Rete Neurale."""
    
    def __init__(self,
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.3,
                 learning_rate: float = 1e-3,
                 batch_size: int = 256,
                 epochs: int = 100,
                 early_stopping_patience: int = 20):
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch non disponibile")
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        self.model = None
        self.input_dim = None
        self.scaler_mean = None
        self.scaler_std = None
        self.is_trained = False
    
    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Addestra la rete neurale."""
        self.input_dim = X.shape[1]
        
        self.model = TradingNet(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        )
        
        self._fit_scaler(X)
        X_scaled = self._transform_scaler(X)
        y_scaled = y
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self._transform_scaler(X_val)
            y_val_scaled = y_val
        else:
            from sklearn.model_selection import train_test_split
            X_scaled, X_val_scaled, y_scaled, y_val_scaled = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_scaled),
            torch.FloatTensor(y_scaled).reshape(-1, 1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val_scaled).reshape(-1, 1)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            
            self.model.eval()
            epoch_val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    epoch_val_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_val_loss = epoch_val_loss / len(val_loader)
            
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.is_trained = True
        
        return {
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch + 1
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice il gradimento."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        self.model.eval()
        X_scaled = self._transform_scaler(X)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled)
            predictions = self.model(X_tensor).numpy().flatten()
        
        return np.clip(predictions, -1, 1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predice le probabilità."""
        predictions = self.predict(X)
        prob_buy = (predictions + 1) / 2
        return np.vstack([1 - prob_buy, prob_buy]).T
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Stima l'importanza delle feature."""
        if not self.is_trained or self.model is None:
            return {}
        
        self.model.eval()
        X_test = torch.randn(1000, self.input_dim)
        X_test.requires_grad = True
        
        outputs = self.model(X_test)
        gradients = torch.autograd.grad(outputs.sum(), X_test, create_graph=True)[0]
        importance = gradients.abs().mean(dim=0).detach().numpy()
        
        return {f'feature_{i}': imp for i, imp in enumerate(importance)}
    
    def _fit_scaler(self, X: np.ndarray):
        """Adatta lo scaler."""
        self.scaler_mean = X.mean(axis=0)
        self.scaler_std = X.std(axis=0)
        self.scaler_std[self.scaler_std < 1e-8] = 1
    
    def _transform_scaler(self, X: np.ndarray) -> np.ndarray:
        """Applica la normalizzazione."""
        return (X - self.scaler_mean) / self.scaler_std