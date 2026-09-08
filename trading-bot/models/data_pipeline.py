# models/data_pipeline.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.decomposition import PCA

class DataPipeline:
    """
    Pipeline di pre-processing per i dati di trading.
    """
    
    def __init__(self,
                 target_column: str = 'future_return',
                 sequence_length: int = 20,
                 use_pca: bool = False,
                 pca_components: int = 20):
        
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.use_pca = use_pca
        self.pca_components = pca_components
        
        self.scaler = RobustScaler()  # Più robusto agli outlier
        self.pca = PCA(n_components=pca_components) if use_pca else None
        
        self.feature_names = []
        self.is_fitted = False
    
    def create_features(self, data: pd.DataFrame, 
                        target_col: str = 'future_return') -> pd.DataFrame:
        """
        Crea feature tecniche da dati OHLCV.
        """
        df = data.copy()
        
        # Feature di prezzo
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'close_vs_sma_{window}'] = (df['close'] / df[f'sma_{window}'] - 1)
        
        # Volatility
        for window in [10, 20, 50]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std() * np.sqrt(252)
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        
        # MACD
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Average True Range
        df['atr'] = self._calculate_atr(df, 14)
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Momentum
        for period in [5, 10, 20, 50]:
            df[f'momentum_{period}'] = (df['close'] / df['close'].shift(period) - 1)
        
        # Target (future return)
        df[target_col] = df['close'].shift(-5) / df['close'] - 1  # 5 giorni
        
        # Rimuovi NaN
        df = df.dropna()
        
        # Seleziona le feature
        feature_cols = [col for col in df.columns if col not in ['close', 'volume', target_col]]
        self.feature_names = feature_cols
        
        return df[feature_cols + [target_col]]
    
    def prepare_data(self, df: pd.DataFrame, 
                     target_col: str = 'future_return') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara i dati per il modello.
        """
        # Separa feature e target
        X = df[df.columns[df.columns != target_col]].values
        y = df[target_col].values
        
        # Scala i dati
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        
        # PCA
        if self.use_pca and self.pca is not None:
            X_scaled = self.pca.fit_transform(X_scaled)
        
        return X_scaled, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Trasforma nuovi dati.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted")
        
        X = df[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        
        if self.use_pca and self.pca is not None:
            X_scaled = self.pca.transform(X_scaled)
        
        return X_scaled
    
    def prepare_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea sequenze per modelli LSTM.
        """
        sequences_X, sequences_y = [], []
        
        for i in range(len(X) - self.sequence_length):
            sequences_X.append(X[i:i + self.sequence_length])
            sequences_y.append(y[i + self.sequence_length])
        
        return np.array(sequences_X), np.array(sequences_y)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcola l'RSI"""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcola l'Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift()
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(period).mean()
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names.copy()