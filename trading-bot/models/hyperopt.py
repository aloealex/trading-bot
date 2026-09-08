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
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
                
                val_pred = model.predict(X_val)
                score = r2_score(y_val, val_pred)
                scores.append(score)
            
            return np.mean(scores)
        
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
                score = r2_score(y_val, val_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        return self.best_params
    
    def optimize_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Ottimizza i pesi dell'ensemble.
        """
        # Predizioni dei modelli base
        from .xgboost_model import XGBoostTradingModel
        from .neural_network import NeuralTradingModel
        
        models = [
            XGBoostTradingModel(n_estimators=500),
            NeuralTradingModel(hidden_dims=[256, 128])
        ]
        
        for model in models:
            model.train(X, y)
        
        predictions = np.column_stack([m.predict(X) for m in models])
        
        # Ottimizza i pesi
        def objective(weights):
            ensemble_pred = np.dot(predictions, weights)
            return -r2_score(y, ensemble_pred)
        
        # Vincoli: somma pesi = 1, pesi >= 0
        from scipy.optimize import minimize
        
        initial_weights = np.array([0.5, 0.5])
        bounds = [(0, 1) for _ in range(len(models))]
        constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        result = minimize(objective, initial_weights, 
                         bounds=bounds, constraints=constraint)
        
        self.best_params = dict(zip(['xgboost', 'neural'], result.x))
        self.best_score = -result.fun
        
        return self.best_params
    
    def get_best_params(self) -> Dict[str, Any]:
        return self.best_params