# models/hyperopt.py
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

class HyperparameterOptimizer:
    """
    Ottimizzazione degli iperparametri con Optuna.
    """
    
    def __init__(self, 
                 n_trials: int = 100,
                 cv_folds: int = 5,
                 random_state: int = 42):
        
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_params = {}
        self.best_score = -np.inf
        self.study = None
    
    def optimize_xgboost(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Ottimizza iperparametri per XGBoost.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna non disponibile")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
            }
            
            # Cross-validation con split temporali
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Addestra e valuta
                import xgboost as xgb
                # xgboost >= 2: early_stopping_rounds va nel costruttore
                model = xgb.XGBRegressor(**params, early_stopping_rounds=50, verbosity=0,
                                         random_state=self.random_state)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
                val_pred = model.predict(X_val)
                # correlazione predizione/rendimento: più significativa dell'R² per il trading
                score = np.corrcoef(val_pred, y_val)[0, 1] if np.std(val_pred) > 0 else 0.0
                scores.append(score)
            
            return float(np.mean(scores))
        
        # Crea lo studio
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return self.best_params

    def optimize_neural_network(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Ottimizza iperparametri per Rete Neurale.
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna non disponibile")
        
        def objective(trial):
            params = {
                'hidden_dims': [
                    trial.suggest_int('hidden_1', 64, 512, step=64),
                    trial.suggest_int('hidden_2', 32, 256, step=32),
                    trial.suggest_int('hidden_3', 16, 128, step=16)
                ],
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'epochs': 100
            }
            
            # Cross-validation con split temporali
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Crea e addestra il modello
                from .neural_network import NeuralTradingModel
                model = NeuralTradingModel(
                    hidden_dims=params['hidden_dims'],
                    dropout_rate=params['dropout_rate'],
                    learning_rate=params['learning_rate'],
                    batch_size=params['batch_size'],
                    epochs=params['epochs']
                )
                
                model.train(X_train, y_train, X_val, y_val)
                val_pred = model.predict(X_val)
                score = np.corrcoef(val_pred, y_val)[0, 1] if np.std(val_pred) > 0 else 0.0
                scores.append(score)
            
            return float(np.mean(scores))
        
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=self.n_trials)
        
        p = self.study.best_params
        # converte i parametri "piatti" di Optuna nel formato del costruttore
        self.best_params = {
            'hidden_dims': [p['hidden_1'], p['hidden_2'], p['hidden_3']],
            'dropout_rate': p['dropout_rate'],
            'learning_rate': p['learning_rate'],
            'batch_size': p['batch_size'],
        }
        self.best_score = self.study.best_value
        
        return self.best_params
    
    def get_best_params(self) -> Dict[str, Any]:
        return self.best_params