# backtest/metrics.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

class BacktestMetrics:
    """
    Calcolo delle metriche di performance per il backtest.
    """
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
        """Calcola lo Sharpe Ratio."""
        if returns.empty or returns.std() == 0:
            return 0
        excess_returns = returns - risk_free_rate / 252
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
        """Calcola il Sortino Ratio."""
        if returns.empty:
            return 0
        downside_returns = returns[returns < 0]
        if downside_returns.empty or downside_returns.std() == 0:
            return 0
        excess_returns = returns.mean() - risk_free_rate / 252
        downside_deviation = downside_returns.std() * np.sqrt(252)
        return (excess_returns * 252) / downside_deviation if downside_deviation > 0 else 0
    
    @staticmethod
    def calmar_ratio(returns: pd.Series, max_drawdown: float) -> float:
        """Calcola il Calmar Ratio."""
        if returns.empty or max_drawdown == 0:
            return 0
        annual_return = (1 + returns.mean()) ** 252 - 1
        return annual_return / abs(max_drawdown)
    
    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> Dict[str, float]:
        """Calcola il drawdown massimo."""
        if equity_curve.empty:
            return {'max_drawdown': 0, 'duration': 0}
        
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        
        max_drawdown = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # Durata del drawdown
        duration = 0
        for i in range(drawdown.index.get_loc(max_dd_idx), -1, -1):
            if drawdown.iloc[i] == 0:
                duration = drawdown.index.get_loc(max_dd_idx) - i
                break
        
        return {'max_drawdown': max_drawdown, 'duration': duration}
    
    @staticmethod
    def win_rate(trades: List) -> float:
        """Calcola il win rate."""
        if not trades:
            return 0
        wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
        return wins / len(trades)
    
    @staticmethod
    def profit_factor(trades: List) -> float:
        """Calcola il profit factor."""
        if not trades:
            return 0
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        return gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    @staticmethod
    def expectancy(trades: List) -> float:
        """Calcola l'expectancy."""
        if not trades:
            return 0
        return np.mean([t.get('pnl', 0) for t in trades])
    
    @staticmethod
    def var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calcola il Value at Risk."""
        if returns.empty:
            return 0
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calcola il Conditional VaR."""
        if returns.empty:
            return 0
        var = BacktestMetrics.var(returns, confidence)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_all_metrics(equity_curve: pd.Series, trades: List) -> Dict[str, Any]:
        """Calcola tutte le metriche."""
        returns = equity_curve.pct_change().dropna()
        
        if equity_curve.empty or returns.empty:
            return {}
        
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
        annual_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        drawdown_metrics = BacktestMetrics.max_drawdown(equity_curve)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': BacktestMetrics.sharpe_ratio(returns),
            'sortino_ratio': BacktestMetrics.sortino_ratio(returns),
            'calmar_ratio': BacktestMetrics.calmar_ratio(returns, drawdown_metrics['max_drawdown']),
            'max_drawdown': drawdown_metrics['max_drawdown'],
            'max_drawdown_duration': drawdown_metrics['duration'],
            'win_rate': BacktestMetrics.win_rate(trades),
            'profit_factor': BacktestMetrics.profit_factor(trades),
            'expectancy': BacktestMetrics.expectancy(trades),
            'var_95': BacktestMetrics.var(returns, 0.95),
            'cvar_95': BacktestMetrics.cvar(returns, 0.95),
            'total_trades': len(trades)
        }