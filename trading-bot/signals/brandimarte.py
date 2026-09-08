# signals/brandimarte.py
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Metriche di rischio calcolate con Monte Carlo"""
    volatility: float
    var_95: float
    cvar_95: float
    var_99: float
    cvar_99: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    probability_of_loss: float
    expected_shortfall: float
    confidence_interval: Tuple[float, float]

class MonteCarloRisk:
    """
    Simulazioni Monte Carlo per la stima del rischio.
    """
    
    def __init__(self,
                 simulations: int = 10000,
                 horizon: int = 252,
                 confidence: float = 0.95,
                 use_bootstrap: bool = True):
        self.simulations = simulations
        self.horizon = horizon
        self.confidence = confidence
        self.use_bootstrap = use_bootstrap
    
    def _estimate_parameters(self, returns: pd.Series) -> Tuple[float, float]:
        """Stima i parametri del moto browniano geometrico."""
        if len(returns) < 10:
            return 0.0, 0.0
        
        mu = returns.mean() * 252
        sigma = returns.std() * np.sqrt(252)
        
        return mu, sigma
    
    def _simulate_gbm(self,
                     current_price: float,
                     mu: float,
                     sigma: float,
                     horizon: int) -> np.ndarray:
        """Simula il moto browniano geometrico."""
        dt = 1 / 252
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        log_returns = np.random.normal(
            drift,
            diffusion,
            (self.simulations, horizon)
        )
        
        cumulative_log = np.cumsum(log_returns, axis=1)
        future_prices = current_price * np.exp(cumulative_log)
        
        return future_prices
    
    def _simulate_bootstrap(self,
                           returns: pd.Series,
                           current_price: float,
                           horizon: int) -> np.ndarray:
        """Simula usando bootstrap dei rendimenti storici."""
        if len(returns) < 20:
            return np.zeros((self.simulations, horizon))
        
        idx = np.random.randint(0, len(returns), (self.simulations, horizon))
        sampled_returns = returns.iloc[idx].values
        
        cumulative_returns = np.cumprod(1 + sampled_returns, axis=1)
        future_prices = current_price * cumulative_returns
        
        return future_prices
    
    def calculate_metrics(self, prices: pd.Series) -> Optional[RiskMetrics]:
        """
        Calcola tutte le metriche di rischio.
        """
        if len(prices) < 30:
            return None
        
        returns = prices.pct_change().dropna()
        current_price = prices.iloc[-1]
        
        if len(returns) < 20:
            return None
        
        mu, sigma = self._estimate_parameters(returns)
        
        if self.use_bootstrap and len(returns) > 100:
            future_prices = self._simulate_bootstrap(returns, current_price, self.horizon)
        else:
            future_prices = self._simulate_gbm(current_price, mu, sigma, self.horizon)
        
        future_returns = (future_prices[:, -1] / current_price - 1)
        
        expected_return = future_returns.mean()
        volatility = future_returns.std() * np.sqrt(252 / self.horizon)
        
        sorted_returns = np.sort(future_returns)
        
        var_95_idx = int((1 - self.confidence) * self.simulations)
        var_95 = sorted_returns[var_95_idx]
        cvar_95 = sorted_returns[:var_95_idx].mean()
        
        var_99_idx = int((1 - 0.99) * self.simulations)
        var_99 = sorted_returns[var_99_idx]
        cvar_99 = sorted_returns[:var_99_idx].mean()
        
        max_drawdown = self._calculate_historical_drawdown(prices)
        
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (returns.mean() * 252) / downside_deviation if downside_deviation > 0 else 0
        
        calmar = (returns.mean() * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        probability_of_loss = (future_returns < 0).mean()
        expected_shortfall = future_returns[future_returns < 0].mean()
        
        ci_lower = np.percentile(future_returns, 2.5)
        ci_upper = np.percentile(future_returns, 97.5)
        
        return RiskMetrics(
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            var_99=var_99,
            cvar_99=cvar_99,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            probability_of_loss=probability_of_loss,
            expected_shortfall=expected_shortfall,
            confidence_interval=(ci_lower, ci_upper)
        )
    
    def _calculate_historical_drawdown(self, prices: pd.Series) -> float:
        """Calcola il drawdown massimo storico."""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max - 1)
        return drawdown.min()
    
    def calculate(self, prices: pd.Series) -> Optional[float]:
        """
        Calcola un punteggio di rischio (più basso = meno rischioso).
        """
        metrics = self.calculate_metrics(prices)
        
        if metrics is None:
            return None
        
        risk_score = -(
            metrics.volatility / 0.5 +
            abs(metrics.var_95) / 0.3 +
            abs(metrics.max_drawdown) / 0.5
        ) / 3
        
        return np.clip(risk_score, -1, 1)
    
    def explain(self, prices: pd.Series) -> Dict[str, Any]:
        """Spiega le metriche di rischio."""
        metrics = self.calculate_metrics(prices)
        
        if metrics is None:
            return {'error': 'Impossibile calcolare le metriche'}
        
        return {
            'volatility': metrics.volatility,
            'var_95': metrics.var_95,
            'cvar_95': metrics.cvar_95,
            'var_99': metrics.var_99,
            'cvar_99': metrics.cvar_99,
            'max_drawdown': metrics.max_drawdown,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'calmar_ratio': metrics.calmar_ratio,
            'probability_of_loss': metrics.probability_of_loss,
            'expected_shortfall': metrics.expected_shortfall,
            'confidence_interval': metrics.confidence_interval,
            'interpretation': self._interpret_metrics(metrics)
        }
    
    def _interpret_metrics(self, metrics: RiskMetrics) -> list:
        """Interpreta le metriche di rischio."""
        interpretation = []
        
        if metrics.volatility < 0.15:
            interpretation.append('Bassa volatilità (meno rischioso)')
        elif metrics.volatility < 0.30:
            interpretation.append('Volatilità moderata')
        else:
            interpretation.append('Alta volatilità (più rischioso)')
        
        if metrics.sharpe_ratio > 1.0:
            interpretation.append('Ottimo Sharpe Ratio (>1.0)')
        elif metrics.sharpe_ratio > 0.5:
            interpretation.append('Buono Sharpe Ratio (>0.5)')
        else:
            interpretation.append('Sharpe Ratio basso')
        
        if abs(metrics.max_drawdown) < 0.10:
            interpretation.append('Drawdown contenuto (<10%)')
        elif abs(metrics.max_drawdown) < 0.25:
            interpretation.append('Drawdown moderato (<25%)')
        else:
            interpretation.append('Drawdown significativo (>25%)')
        
        if abs(metrics.var_95) < 0.05:
            interpretation.append('VaR 95% contenuto (<5%)')
        else:
            interpretation.append(f'VaR 95%: {abs(metrics.var_95):.1%}')
        
        if metrics.probability_of_loss < 0.40:
            interpretation.append('Bassa probabilità di perdita')
        elif metrics.probability_of_loss < 0.50:
            interpretation.append('Probabilità di perdita moderata')
        else:
            interpretation.append('Alta probabilità di perdita')
        
        return interpretation