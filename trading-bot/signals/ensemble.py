# signals/ensemble.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .clenow import ClenowMomentum
from .raschke import RaschkePatterns
from .brandimarte import MonteCarloRisk, RiskMetrics

logger = logging.getLogger(__name__)

@dataclass
class EnsembleResult:
    """Risultato dell'ensemble dei segnali"""
    symbol: str
    timestamp: datetime
    gradimento: float
    confidence: float
    components: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Optional[RiskMetrics] = None
    signals: Dict[str, Any] = field(default_factory=dict)
    interpretation: List[str] = field(default_factory=list)

class SignalEnsemble:
    """
    Combina i segnali dai tre approcci in un unico "gradimento" finale.
    """
    
    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 adaptive_weights: bool = True):
        
        self.weights = weights or {
            'momentum': 0.40,
            'raschke': 0.25,
            'risk': -0.20,
            'mean_reversion': 0.15
        }
        
        self.adaptive_weights = adaptive_weights
        self.performance_history = []
        
        self.clenow = ClenowMomentum()
        self.raschke = RaschkePatterns()
        self.risk = MonteCarloRisk()
    
    def calculate(self, prices: pd.Series, symbol: str = '') -> EnsembleResult:
        """Calcola il gradimento finale per un singolo simbolo."""
        if len(prices) < 200:
            return EnsembleResult(
                symbol=symbol,
                timestamp=datetime.now(),
                gradimento=0.0,
                confidence=0.0,
                components={},
                interpretation=['Dati insufficienti']
            )
        
        # 1. Clenow Momentum
        momentum = self.clenow.calculate(prices) or 0.0
        normalized_momentum = np.clip(momentum / 50, -1, 1)
        
        # 2. Raschke Patterns
        raschke_signal = self.raschke.calculate(prices) or 0.0
        
        # 3. Risk Metrics
        risk_metrics = self.risk.calculate_metrics(prices)
        if risk_metrics:
            risk_score = -np.clip(risk_metrics.volatility / 0.5, 0, 1)
            sharpe_score = np.clip(risk_metrics.sharpe_ratio / 2, 0, 1)
            risk_score = (risk_score + sharpe_score) / 2
        else:
            risk_score = 0.0
        
        # 4. Mean Reversion
        er = self._calculate_efficiency_ratio(prices)
        mean_reversion_score = np.clip(1 - er * 3, -0.5, 0.5)
        
        # 5. RSI
        rsi = self._calculate_rsi(prices, 14)
        rsi_score = (rsi - 50) / 50 if rsi else 0
        
        # 6. Gradimento finale
        gradimento = (
            self.weights['momentum'] * normalized_momentum +
            self.weights['raschke'] * raschke_signal +
            self.weights['risk'] * risk_score +
            self.weights['mean_reversion'] * mean_reversion_score +
            0.05 * rsi_score
        )
        
        gradimento = np.clip(gradimento, -1, 1)
        
        # 7. Confidenza
        confidence = self._calculate_confidence(
            normalized_momentum,
            raschke_signal,
            risk_score,
            mean_reversion_score
        )
        
        # 8. Interpretazione
        interpretation = self._interpret_signal(
            gradimento,
            normalized_momentum,
            raschke_signal,
            risk_metrics
        )
        
        return EnsembleResult(
            symbol=symbol,
            timestamp=datetime.now(),
            gradimento=gradimento,
            confidence=confidence,
            components={
                'momentum': normalized_momentum,
                'raschke': raschke_signal,
                'risk_score': risk_score,
                'mean_reversion': mean_reversion_score,
                'rsi_score': rsi_score
            },
            risk_metrics=risk_metrics,
            signals={
                'clenow': momentum,
                'raschke_components': self.raschke.calculate_components(prices),
                'raw_momentum': momentum
            },
            interpretation=interpretation
        )
    
    def calculate_batch(self, prices_dict: Dict[str, pd.Series]) -> Dict[str, EnsembleResult]:
        """Calcola il gradimento per un batch di simboli."""
        results = {}
        for symbol, prices in prices_dict.items():
            results[symbol] = self.calculate(prices, symbol)
        return results
    
    def rank_symbols(self, prices_dict: Dict[str, pd.Series]) -> pd.DataFrame:
        """Classifica i simboli per gradimento."""
        results = self.calculate_batch(prices_dict)
        
        data = []
        for symbol, result in results.items():
            data.append({
                'symbol': symbol,
                'gradimento': result.gradimento,
                'confidence': result.confidence,
                'momentum': result.components.get('momentum', 0),
                'raschke': result.components.get('raschke', 0),
                'risk_score': result.components.get('risk_score', 0),
                'mean_reversion': result.components.get('mean_reversion', 0)
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('gradimento', ascending=False).reset_index(drop=True)
        df['rank'] = df.index + 1
        
        return df
    
    def _calculate_efficiency_ratio(self, prices: pd.Series, window: int = 20) -> float:
        """Calcola l'Efficiency Ratio di Kaufman."""
        if len(prices) < window:
            return 0.0
        
        returns = prices.diff()
        net_move = abs(prices.iloc[-1] - prices.iloc[-window])
        sum_move = returns.iloc[-window:].abs().sum()
        
        if sum_move == 0:
            return 0.0
        
        return net_move / sum_move
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calcola l'RSI."""
        if len(prices) < period + 1:
            return None
        
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        if avg_loss.iloc[-1] == 0:
            return 100.0
        
        rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_confidence(self, *signal_values: float) -> float:
        """Calcola la confidenza del segnale."""
        signs = np.sign(signal_values)
        positive_count = np.sum(signs > 0)
        negative_count = np.sum(signs < 0)
        
        if positive_count == len(signal_values) or negative_count == len(signal_values):
            return 0.9
        if abs(positive_count - negative_count) >= 2:
            return 0.7
        if positive_count > 0 and negative_count > 0:
            return 0.5
        
        return 0.3
    
    def _interpret_signal(self,
                         gradimento: float,
                         momentum: float,
                         raschke: float,
                         risk_metrics: Optional[RiskMetrics]) -> List[str]:
        """Interpreta il segnale generato."""
        interpretation = []
        
        if gradimento > 0.6:
            interpretation.append('🚀 Segnale di ACQUISTO FORTE')
        elif gradimento > 0.3:
            interpretation.append('📈 Segnale di ACQUISTO')
        elif gradimento < -0.6:
            interpretation.append('🔻 Segnale di VENDITA FORTE')
        elif gradimento < -0.3:
            interpretation.append('📉 Segnale di VENDITA')
        else:
            interpretation.append('⏸️ Segnale NEUTRO')
        
        if momentum > 0.5:
            interpretation.append(f'   Momentum positivo: {momentum:.2f}')
        elif momentum < -0.5:
            interpretation.append(f'   Momentum negativo: {momentum:.2f}')
        
        if raschke > 0.3:
            interpretation.append('   Pattern di acquisto (Raschke)')
        elif raschke < -0.3:
            interpretation.append('   Pattern di vendita (Raschke)')
        
        if risk_metrics:
            if risk_metrics.volatility < 0.15:
                interpretation.append(f'   Rischio basso (vol: {risk_metrics.volatility:.1%})')
            elif risk_metrics.volatility > 0.35:
                interpretation.append(f'   Rischio alto (vol: {risk_metrics.volatility:.1%})')
            
            if risk_metrics.sharpe_ratio > 1.0:
                interpretation.append(f'   Sharpe Ratio > 1.0 ({risk_metrics.sharpe_ratio:.2f})')
        
        return interpretation
    
    def get_trading_signal(self, result: EnsembleResult) -> Dict[str, Any]:
        """Converte il gradimento in un segnale di trading."""
        gradimento = result.gradimento
        confidence = result.confidence
        
        if gradimento > 0.6:
            action = 'STRONG_BUY'
            priority = 5
        elif gradimento > 0.3:
            action = 'BUY'
            priority = 4
        elif gradimento < -0.6:
            action = 'STRONG_SELL'
            priority = 5
        elif gradimento < -0.3:
            action = 'SELL'
            priority = 4
        else:
            action = 'HOLD'
            priority = 1
        
        position_size = abs(gradimento) * confidence
        
        return {
            'symbol': result.symbol,
            'action': action,
            'priority': priority,
            'gradimento': gradimento,
            'confidence': confidence,
            'position_size': position_size,
            'interpretation': result.interpretation,
            'components': result.components
        }