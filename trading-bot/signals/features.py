# signals/features.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from .clenow import ClenowMomentum
from .raschke import RaschkePatterns
from .brandimarte import MonteCarloRisk

class FeatureEngineer:
    """
    Ingegneria delle feature per il modello di Machine Learning.
    """
    
    def __init__(self):
        self.clenow = ClenowMomentum()
        self.raschke = RaschkePatterns()
        self.risk = MonteCarloRisk()
    
    def extract_features(self, prices: pd.Series) -> Dict[str, float]:
        """
        Estrae tutte le feature per un singolo simbolo.
        """
        if len(prices) < 200:
            return {}
        
        returns = prices.pct_change().dropna()
        if len(returns) < 50:
            return {}
        
        features = {}
        
        # 1. Feature di momentum
        features['momentum_10d'] = (prices.iloc[-1] / prices.iloc[-10] - 1) * 100
        features['momentum_20d'] = (prices.iloc[-1] / prices.iloc[-20] - 1) * 100
        features['momentum_50d'] = (prices.iloc[-1] / prices.iloc[-50] - 1) * 100
        features['momentum_125d'] = (prices.iloc[-1] / prices.iloc[-125] - 1) * 100
        
        # 2. Moving Averages
        features['sma_20'] = prices.rolling(20).mean().iloc[-1]
        features['sma_50'] = prices.rolling(50).mean().iloc[-1]
        features['sma_200'] = prices.rolling(200).mean().iloc[-1]
        features['price_vs_sma_20'] = (prices.iloc[-1] / features['sma_20'] - 1) * 100
        features['price_vs_sma_50'] = (prices.iloc[-1] / features['sma_50'] - 1) * 100
        features['price_vs_sma_200'] = (prices.iloc[-1] / features['sma_200'] - 1) * 100
        
        # 3. Volatility
        features['volatility_20d'] = returns.iloc[-20:].std() * np.sqrt(252)
        features['volatility_50d'] = returns.iloc[-50:].std() * np.sqrt(252)
        features['volatility_ratio'] = features['volatility_20d'] / features['volatility_50d']
        
        # 4. Volume (se disponibile, altrimenti placeholder)
        # features['volume_ratio'] = ...
        
        # 5. RSI
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        if avg_loss.iloc[-1] > 0:
            rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
            features['rsi_14'] = 100 - (100 / (1 + rs))
        else:
            features['rsi_14'] = 100
        
        # 6. Efficiency Ratio
        net_move = abs(prices.iloc[-1] - prices.iloc[-20])
        sum_move = abs(returns.iloc[-20:]).sum()
        features['efficiency_ratio'] = net_move / sum_move if sum_move > 0 else 0
        
        # 7. Clenow Momentum Score
        features['clenow_score'] = self.clenow.calculate(prices) or 0
        
        # 8. Raschke Signal
        features['raschke_signal'] = self.raschke.calculate(prices) or 0
        
        # 9. Risk Metrics
        risk_metrics = self.risk.calculate_metrics(prices)
        if risk_metrics:
            features['risk_volatility'] = risk_metrics.volatility
            features['risk_var_95'] = risk_metrics.var_95
            features['risk_sharpe'] = risk_metrics.sharpe_ratio
            features['risk_drawdown'] = risk_metrics.max_drawdown
            features['risk_loss_prob'] = risk_metrics.probability_of_loss
        
        # 10. Price Patterns
        # Higher High / Lower Low
        features['higher_high'] = 1 if prices.iloc[-1] > prices.iloc[-10:].max() else 0
        features['lower_low'] = 1 if prices.iloc[-1] < prices.iloc[-10:].min() else 0
        
        # 11. Range
        features['range_10d'] = (prices.iloc[-10:].max() - prices.iloc[-10:].min()) / prices.iloc[-10:].mean()
        features['range_20d'] = (prices.iloc[-20:].max() - prices.iloc[-20:].min()) / prices.iloc[-20:].mean()
        
        # 12. Skewness e Kurtosis dei rendimenti
        features['return_skew'] = returns.iloc[-50:].skew()
        features['return_kurtosis'] = returns.iloc[-50:].kurtosis()
        
        # 13. Drawdown attuale
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        features['current_drawdown'] = (cumulative.iloc[-1] / running_max.iloc[-1] - 1)
        
        return features
    
    def extract_batch(self, prices_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Estrae le feature per un batch di simboli.
        """
        all_features = []
        for symbol, prices in prices_dict.items():
            features = self.extract_features(prices)
            features['symbol'] = symbol
            all_features.append(features)
        
        df = pd.DataFrame(all_features)
        df = df.set_index('symbol')
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """
        Restituisce la lista dei nomi delle feature.
        """
        return [
            'momentum_10d', 'momentum_20d', 'momentum_50d', 'momentum_125d',
            'price_vs_sma_20', 'price_vs_sma_50', 'price_vs_sma_200',
            'volatility_20d', 'volatility_50d', 'volatility_ratio',
            'rsi_14', 'efficiency_ratio',
            'clenow_score', 'raschke_signal',
            'risk_volatility', 'risk_var_95', 'risk_sharpe', 'risk_drawdown', 'risk_loss_prob',
            'higher_high', 'lower_low',
            'range_10d', 'range_20d',
            'return_skew', 'return_kurtosis', 'current_drawdown'
        ]