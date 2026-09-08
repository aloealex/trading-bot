# risk/portfolio_risk.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PortfolioRiskMetrics:
    """Metriche di rischio del portafoglio"""
    total_value: float
    daily_var_95: float
    daily_var_99: float
    daily_cvar_95: float
    daily_cvar_99: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    diversification_ratio: float
    correlation_matrix: pd.DataFrame
    beta_to_market: float

class PortfolioRiskManager:
    """
    Gestione del rischio a livello di portafoglio.
    """
    
    def __init__(self,
                 initial_capital: float = 100000,
                 max_portfolio_risk: float = 0.20,
                 max_concentration: float = 0.25,
                 market_index: Optional[str] = 'SPY'):
        
        self.initial_capital = initial_capital
        self.max_portfolio_risk = max_portfolio_risk
        self.max_concentration = max_concentration
        self.market_index = market_index
        
        self.positions = {}
        self.price_history = {}
        self.returns_history = {}
    
    def update_position(self, symbol: str, quantity: int, price: float):
        """Aggiorna una posizione."""
        self.positions[symbol] = {
            'quantity': quantity,
            'price': price,
            'value': quantity * price
        }
    
    def remove_position(self, symbol: str):
        """Rimuove una posizione."""
        if symbol in self.positions:
            del self.positions[symbol]
    
    def update_price_history(self, symbol: str, prices: pd.Series):
        """Aggiorna lo storico dei prezzi."""
        self.price_history[symbol] = prices
        self.returns_history[symbol] = prices.pct_change().dropna()
    
    def calculate_portfolio_value(self) -> float:
        """Calcola il valore totale del portafoglio."""
        total = 0
        for symbol, pos in self.positions.items():
            total += pos['value']
        return total
    
    def calculate_portfolio_weights(self) -> Dict[str, float]:
        """Calcola i pesi delle posizioni."""
        total = self.calculate_portfolio_value()
        if total == 0:
            return {}
        return {symbol: pos['value'] / total for symbol, pos in self.positions.items()}
    
    def calculate_portfolio_returns(self, period: int = 252) -> pd.Series:
        """
        Calcola i rendimenti storici del portafoglio.
        """
        if not self.positions:
            return pd.Series()
        
        weights = self.calculate_portfolio_weights()
        portfolio_returns = pd.Series(index=self.returns_history.get(list(self.positions.keys())[0], pd.Series()).index)
        
        for symbol, weight in weights.items():
            if symbol in self.returns_history:
                returns = self.returns_history[symbol] * weight
                portfolio_returns = portfolio_returns.add(returns, fill_value=0)
        
        return portfolio_returns
    
    def calculate_risk_metrics(self) -> PortfolioRiskMetrics:
        """
        Calcola le metriche di rischio del portafoglio.
        """
        total_value = self.calculate_portfolio_value()
        
        if total_value == 0 or not self.positions:
            return PortfolioRiskMetrics(
                total_value=0,
                daily_var_95=0,
                daily_var_99=0,
                daily_cvar_95=0,
                daily_cvar_99=0,
                volatility=0,
                sharpe_ratio=0,
                max_drawdown=0,
                current_drawdown=0,
                diversification_ratio=1,
                correlation_matrix=pd.DataFrame(),
                beta_to_market=0
            )
        
        portfolio_returns = self.calculate_portfolio_returns()
        
        if portfolio_returns.empty:
            return PortfolioRiskMetrics(
                total_value=total_value,
                daily_var_95=0,
                daily_var_99=0,
                daily_cvar_95=0,
                daily_cvar_99=0,
                volatility=0,
                sharpe_ratio=0,
                max_drawdown=0,
                current_drawdown=0,
                diversification_ratio=1,
                correlation_matrix=pd.DataFrame(),
                beta_to_market=0
            )
        
        # VaR
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        
        # CVaR
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        cvar_99 = portfolio_returns[portfolio_returns <= var_99].mean()
        
        # Volatilità annualizzata
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe = (portfolio_returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max - 1)
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1] if not drawdown.empty else 0
        
        # Diversification Ratio
        weights = self.calculate_portfolio_weights()
        weighted_vol = 0
        for symbol, weight in weights.items():
            if symbol in self.returns_history:
                weighted_vol += weight * self.returns_history[symbol].std() * np.sqrt(252)
        
        diversification_ratio = weighted_vol / volatility if volatility > 0 else 1
        
        # Correlation Matrix
        returns_list = []
        for symbol in self.positions.keys():
            if symbol in self.returns_history:
                returns_list.append(self.returns_history[symbol].rename(symbol))
        
        if len(returns_list) > 1:
            correlation_matrix = pd.concat(returns_list, axis=1).corr()
        else:
            correlation_matrix = pd.DataFrame()
        
        # Beta to market
        beta = 1
        if self.market_index and self.market_index in self.returns_history:
            market_returns = self.returns_history[self.market_index]
            if len(portfolio_returns) > 0:
                covariance = np.cov(portfolio_returns, market_returns)[0, 1]
                variance = np.var(market_returns)
                beta = covariance / variance if variance > 0 else 1
        
        return PortfolioRiskMetrics(
            total_value=total_value,
            daily_var_95=var_95 * total_value,
            daily_var_99=var_99 * total_value,
            daily_cvar_95=cvar_95 * total_value,
            daily_cvar_99=cvar_99 * total_value,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            diversification_ratio=diversification_ratio,
            correlation_matrix=correlation_matrix,
            beta_to_market=beta
        )
    
    def check_portfolio_limits(self) -> Tuple[bool, List[str]]:
        """
        Verifica che il portafoglio rispetti i limiti di rischio.
        
        Returns:
            Tuple: (limiti_rispettati, lista_violazioni)
        """
        violations = []
        
        # Calcola le metriche
        metrics = self.calculate_risk_metrics()
        weights = self.calculate_portfolio_weights()
        
        # Verifica la concentrazione
        for symbol, weight in weights.items():
            if weight > self.max_concentration:
                violations.append(f"Concentrazione {symbol}: {weight:.1%} > {self.max_concentration:.1%}")
        
        # Verifica il rischio totale
        if metrics.volatility > self.max_portfolio_risk:
            violations.append(f"Volatilità portafoglio: {metrics.volatility:.1%} > {self.max_portfolio_risk:.1%}")
        
        # Verifica il drawdown
        if metrics.current_drawdown < -self.max_portfolio_risk:
            violations.append(f"Drawdown corrente: {metrics.current_drawdown:.1%} < {-self.max_portfolio_risk:.1%}")
        
        # Verifica la diversificazione
        if metrics.diversification_ratio < 0.5:
            violations.append(f"Bassa diversificazione: {metrics.diversification_ratio:.2f}")
        
        return len(violations) == 0, violations
    
    def calculate_position_risk_contribution(self) -> Dict[str, float]:
        """
        Calcola il contributo al rischio di ogni posizione.
        """
        if not self.positions:
            return {}
        
        weights = self.calculate_portfolio_weights()
        portfolio_returns = self.calculate_portfolio_returns()
        
        contributions = {}
        for symbol, weight in weights.items():
            if symbol in self.returns_history:
                returns = self.returns_history[symbol]
                correlation = returns.corr(portfolio_returns) if len(portfolio_returns) > 0 else 0
                volatility = returns.std() * np.sqrt(252)
                contributions[symbol] = weight * correlation * volatility / portfolio_returns.std() if portfolio_returns.std() > 0 else 0
        
        return contributions
    
    def get_stress_test_scenarios(self) -> Dict[str, Dict]:
        """
        Genera scenari di stress test.
        """
        scenarios = {
            'market_crash': {'description': 'Crollo del mercato (-15%)', 'factor': 0.85},
            'volatility_spike': {'description': 'Picco di volatilità (+50%)', 'factor': 1.5},
            'liquidity_crisis': {'description': 'Crisi di liquidità (-20%)', 'factor': 0.80},
            'sector_rotation': {'description': 'Rotazione settoriale (-10%)', 'factor': 0.90}
        }
        
        return scenarios