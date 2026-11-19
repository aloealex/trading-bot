# models/predict.py
"""
Predizione con un modello salvato da ModelTrainer.

Il modello stima il rendimento a N giorni; qui viene convertito in "gradimento"
(-1..+1) con tanh(pred / scala), dove la scala è la deviazione standard delle
predizioni in addestramento. Senza questa conversione le predizioni (~±0,01)
non avrebbero mai superato le soglie di acquisto/vendita (±0,3).
"""
import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .features import build_features

logger = logging.getLogger(__name__)


class ModelPredictor:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.meta: Dict[str, Any] = {}
        self.features = []
        if model_path:
            self.load(model_path)

    @staticmethod
    def exists(path: str) -> bool:
        return os.path.isfile(os.path.join(path, 'model.pkl')) and os.path.isfile(os.path.join(path, 'meta.json'))

    def load(self, path: str) -> None:
        with open(os.path.join(path, 'model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(path, 'meta.json')) as f:
            self.meta = json.load(f)
        self.features = self.meta['features']
        logger.info(f"Modello {self.meta.get('model_type')} caricato da {path} "
                    f"(addestrato il {self.meta.get('trained_at')})")

    def _to_gradimento(self, pred: np.ndarray) -> np.ndarray:
        scale = self.meta.get('prediction_scale') or 1e-3
        return np.tanh(np.asarray(pred, dtype=float) / scale)

    def predict_symbol(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gradimento sull'ultima barra disponibile di un DataFrame OHLCV."""
        if self.model is None:
            return {'gradimento': 0.0, 'confidence': 0.0, 'error': 'Modello non caricato'}
        try:
            feats = build_features(df)[self.features]
            last = feats.iloc[[-1]]
            if last.isnull().any(axis=None):
                missing = last.columns[last.isnull().iloc[0]].tolist()
                return {'gradimento': 0.0, 'confidence': 0.0,
                        'error': f'Storico insufficiente per le feature: {missing}'}
            X = last.values
            if hasattr(self.model, 'predict_with_confidence'):
                pred, conf = self.model.predict_with_confidence(X)
                pred, confidence = float(pred[0]), float(conf[0])
            else:
                pred, confidence = float(self.model.predict(X)[0]), 0.5
            return {
                'gradimento': float(self._to_gradimento(pred)),
                'expected_return': pred,
                'confidence': confidence,
                'timestamp': datetime.now(),
            }
        except Exception as e:
            logger.error(f"Errore nella predizione: {e}")
            return {'gradimento': 0.0, 'confidence': 0.0, 'error': str(e)}

    def predict_series(self, df: pd.DataFrame) -> pd.Series:
        """Gradimento per ogni barra (le feature sono causali: nessun dato futuro)."""
        feats = build_features(df)[self.features].dropna()
        if feats.empty:
            return pd.Series(dtype=float)
        return pd.Series(self._to_gradimento(self.model.predict(feats.values)), index=feats.index)

    def predict_batch(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        return {s: self.predict_symbol(df) for s, df in data.items() if not df.empty}
