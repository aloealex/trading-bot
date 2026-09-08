# data/historical.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from .market_data import MarketData
import logging

logger = logging.getLogger(__name__)

class HistoricalData:
    """
    Gestione e pre-processing dei dati storici.
    """
    
    def __init__(self, market_data: MarketData):
        self.market_data = market_data
        self._cache = {}
    
    def fetch_and_prepare(self,
                         symbols: List[str],
                         lookback_days: int = 252,
                         interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Recupera e prepara i dati per l'analisi.
        """
        data = self.market_data.get_historical_batch(symbols, lookback_days, interval)
        
        # Prepara ogni simbolo
        prepared = {}
        for symbol, df in data.items():
            if not df.empty:
                prepared[symbol] = self._prepare_data(df)
        
        return prepared
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara i dati per l'analisi tecnica.
        """
        df = df.copy()
        
        # Assicura che le colonne siano in minuscolo
        df.columns = df.columns.str.lower()
        
        # Calcola i rendimenti
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatilità
        for window in [10, 20, 50]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std() * np.sqrt(252)
        
        # ATR
        df['atr'] = self._calculate_atr(df)
        
        # SMA
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
        
        # Momentum
        for period in [5, 10, 20, 50]:
            df[f'momentum_{period}'] = (df['close'] / df['close'].shift(period) - 1)
        
        # Drop NaN
        df = df.dropna()
        
        return df
    
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
    
    def get_multi_timeframe_data(self, 
                                 symbols: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Recupera dati su multipli timeframe.
        """
        timeframes = {
            '1m': 30,   # 30 giorni
            '5m': 60,   # 60 giorni
            '15m': 90,  # 90 giorni
            '1h': 180,  # 180 giorni
            '1d': 365,  # 1 anno
            '1w': 730   # 2 anni
        }
        
        result = {}
        
        for symbol in symbols:
            result[symbol] = {}
            for interval, days in timeframes.items():
                df = self.market_data.get_historical_data(symbol, days, interval)
                if not df.empty:
                    result[symbol][interval] = df
        
        return result
    
    def detect_market_regime(self, 
                             data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        Rileva il regime di mercato per ogni simbolo.
        """
        regimes = {}
        
        for symbol, df in data.items():
            if df.empty or 'close' not in df.columns:
                regimes[symbol] = 'unknown'
                continue
            
            close = df['close']
            
            # Calcola l'Efficiency Ratio
            returns = close.pct_change()
            er = self._calculate_efficiency_ratio(close)
            
            # Trend direction
            sma_50 = close.rolling(50).mean()
            sma_200 = close.rolling(200).mean()
            trend_direction = 'up' if close.iloc[-1] > sma_50.iloc[-1] else 'down'
            trend_strength = abs(close.iloc[-1] / sma_50.iloc[-1] - 1)
            
            # Volatility
            volatility = returns.iloc[-20:].std() * np.sqrt(252)
            volatility_regime = 'high' if volatility > 0.3 else 'low'
            
            # Determina il regime
            if er > 0.6 and trend_strength > 0.05:
                regimes[symbol] = f'strong_{trend_direction}_trend'
            elif er > 0.4:
                regimes[symbol] = f'{trend_direction}_trend'
            elif er < 0.3:
                regimes[symbol] = 'range_bound'
            else:
                regimes[symbol] = 'choppy'
            
            # Aggiungi volatilità al regime
            if volatility_regime == 'high':
                regimes[symbol] += '_high_vol'
        
        return regimes
    
    def _calculate_efficiency_ratio(self, prices: pd.Series, window: int = 20) -> float:
        """Calcola l'Efficiency Ratio"""
        if len(prices) < window:
            return 0.0
        
        returns = prices.diff()
        net_move = abs(prices.iloc[-1] - prices.iloc[-window])
        sum_move = returns.iloc[-window:].abs().sum()
        
        if sum_move == 0:
            return 0.0
        
        return net_move / sum_move