# models/train.py
"""
Addestramento del modello ML.

Uso tipico:
    trainer = ModelTrainer(model_type='xgboost')
    metrics = trainer.train(data)          # data: dict simbolo -> DataFrame OHLCV
    trainer.save('./saved_models/xgboost')
"""
import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .features import FEATURE_NAMES, TARGET_HORIZON, build_dataset

logger = logging.getLogger(__name__)


def create_model(model_type: str, params: Optional[Dict] = None, feature_names: Optional[List[str]] = None):
    params = dict(params or {})
    if model_type == 'xgboost':
        from .xgboost_model import XGBoostTradingModel
        return XGBoostTradingModel(feature_names=feature_names, **params)
    if model_type == 'neural':
        from .neural_network import NeuralTradingModel
        return NeuralTradingModel(feature_names=feature_names, **params)
    if model_type == 'ensemble':
        from .ensemble_model import EnsembleModel
        return EnsembleModel(feature_names=feature_names, **params)
    raise ValueError(f"Tipo di modello non supportato: {model_type} (usa xgboost, neural, ensemble)")


class ModelTrainer:
    """Addestramento e salvataggio dei modelli."""

    def __init__(self,
                 model_type: str = 'xgboost',
                 model_params: Optional[Dict] = None,
                 horizon: int = TARGET_HORIZON,
                 features: Optional[List[str]] = None):
        self.model_type = model_type
        self.model_params = model_params or {}
        self.horizon = horizon
        self.features = list(features or FEATURE_NAMES)
        self.model = None
        self.meta: Dict[str, Any] = {}
        self.is_trained = False

    def prepare_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        return build_dataset(data, self.horizon)

    def train(self, data: Dict[str, pd.DataFrame], test_fraction: float = 0.2) -> Dict[str, Any]:
        """
        Addestra con split TEMPORALE per data (uguale per tutti i simboli):
        train = passato, test = periodo più recente mai visto dal modello.
        Tra i due c'è un "buco" di `horizon` giorni, perché il target guarda avanti di
        `horizon` giorni e altrimenti l'ultima parte del train conoscerebbe il test.
        """
        dataset = self.prepare_data(data)
        if len(dataset) < 200:
            raise ValueError(f"Dati insufficienti per l'addestramento ({len(dataset)} righe)")

        dates = np.array(sorted(dataset.index.unique()))
        split_i = int(len(dates) * (1 - test_fraction))
        test_start = dates[split_i]
        train_end = dates[max(0, split_i - self.horizon)]
        train = dataset[dataset.index < train_end]
        test = dataset[dataset.index >= test_start]

        X_train, y_train = train[self.features].values, train['target'].values
        X_test, y_test = test[self.features].values, test['target'].values

        self.model = create_model(self.model_type, self.model_params, self.features)
        train_metrics = self.model.train(X_train, y_train)

        # scala per convertire il rendimento atteso in gradimento (-1..+1)
        pred_train = self.model.predict(X_train)
        pred_scale = float(np.std(pred_train)) or 1e-3

        pred_test = self.model.predict(X_test)
        test_metrics = self._evaluate(pred_test, y_test)

        self.meta = {
            'model_type': self.model_type,
            'features': self.features,
            'horizon': self.horizon,
            'prediction_scale': pred_scale,
            'trained_at': datetime.now().isoformat(timespec='seconds'),
            'train_period': [str(train.index.min().date()), str(train.index.max().date())],
            'test_period': [str(test.index.min().date()), str(test.index.max().date())],
            'symbols': sorted(dataset['symbol'].unique().tolist()),
            'n_train': int(len(train)),
            'n_test': int(len(test)),
            'train_metrics': _jsonable(train_metrics),
            'test_metrics': test_metrics,
        }
        self.is_trained = True
        logger.info(f"Modello {self.model_type} addestrato: {test_metrics}")
        return self.meta

    @staticmethod
    def _evaluate(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Metriche fuori campione (periodo di test)."""
        if len(y) == 0:
            return {}
        corr = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 else 0.0
        rank_ic = float(pd.Series(pred).corr(pd.Series(y), method='spearman')) if np.std(pred) > 0 else 0.0
        hit = float((np.sign(pred) == np.sign(y)).mean())
        top = y[pred > np.quantile(pred, 0.8)]
        bottom = y[pred < np.quantile(pred, 0.2)]
        return {
            'correlazione': corr,
            'rank_ic': rank_ic,
            'direzione_corretta': hit,
            'rend_medio_top20%': float(top.mean()) if len(top) else 0.0,
            'rend_medio_bottom20%': float(bottom.mean()) if len(bottom) else 0.0,
            'rend_medio_tutti': float(y.mean()),
        }

    def save(self, path: str) -> None:
        if not self.is_trained:
            raise ValueError("Modello non addestrato")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        with open(os.path.join(path, 'meta.json'), 'w') as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)
        logger.info(f"Modello salvato in {path}")

    @classmethod
    def load(cls, path: str) -> 'ModelTrainer':
        with open(os.path.join(path, 'meta.json')) as f:
            meta = json.load(f)
        trainer = cls(model_type=meta['model_type'], horizon=meta['horizon'], features=meta['features'])
        with open(os.path.join(path, 'model.pkl'), 'rb') as f:
            trainer.model = pickle.load(f)
        trainer.meta = meta
        trainer.is_trained = True
        return trainer

    def get_model_summary(self) -> Dict[str, Any]:
        return {
            **self.meta,
            'feature_importance': self.model.get_feature_importance() if self.model else {},
        }


def _jsonable(d: Dict) -> Dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _jsonable(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out
