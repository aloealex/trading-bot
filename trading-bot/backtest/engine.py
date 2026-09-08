# backtest/engine.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Rappresenta una singola operazione di trading."""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str
    pnl: float
    pnl_percent: float
    holding_days: int
    max_favorable_excursion: float
    max_adverse_excursion: float
    exit_reason: str

@dataclass
class BacktestResult:
    """Risultati completi del backtest."""
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    avg_holding_days: float
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    trades: List[Trade]
    monthly_returns: pd.DataFrame
    yearly_returns: pd.DataFrame
    var_95: float
    cvar_95: float

class BacktestEngine:
    """
    Motore di backtesting professionale.
    """
    
    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 max_positions: int = 10,
                 position_sizing: str = 'kelly',
                 risk_per_trade: float = 0.02):
        
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_positions = max_positions
        self.position_sizing = position_sizing
        self.risk_per_trade = risk_per_trade
        
        self.results = None
        self.equity_curve = []
        self.trades = []
        self.positions = {}
        self.capital = initial_capital
    
    def run(self,
            data: Dict[str, pd.DataFrame],
            signals: Dict[str, pd.Series],
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None,
            trailing_stop: Optional[float] = None,
            max_holding_days: Optional[int] = None) -> BacktestResult:
        """Esegue il backtest completo."""
        self.capital = self.initial_capital
        self.equity_curve = [self.capital]
        self.trades = []
        self.positions = {}
        
        dates = self._align_dates(data)
        
        for i, date in enumerate(dates):
            # 1. Chiudi le posizioni
            self._close_positions(date, data, stop_loss, take_profit, trailing_stop, max_holding_days)
            
            # 2. Genera i segnali per il giorno
            daily_signals = self._get_daily_signals(signals, date)
            
            # 3. Apri nuove posizioni
            self._open_positions(date, data, daily_signals)
            
            # 4. Aggiorna l'equity
            self.capital = self._update_equity(date, data)
            self.equity_curve.append(self.capital)
        
        # Calcola le metriche
        self.results = self._calculate_metrics()
        
        return self.results
    
    def _align_dates(self, data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """Allinea le date tra tutti i simboli."""
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        return sorted(all_dates)
    
    def _get_daily_signals(self, signals: Dict[str, pd.Series], date: datetime) -> Dict[str, float]:
        """Ottiene i segnali per un giorno specifico."""
        daily_signals = {}
        for symbol, signal_series in signals.items():
            if date in signal_series.index:
                signal = signal_series.loc[date]
                if abs(signal) > 0.1:
                    daily_signals[symbol] = signal
        return daily_signals
    
    def _open_positions(self, date: datetime, data: Dict[str, pd.DataFrame],
                        signals: Dict[str, float]):
        """Apre nuove posizioni basate sui segnali."""
        if len(self.positions) >= self.max_positions:
            return
        
        sorted_signals = sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for symbol, signal in sorted_signals:
            if symbol in self.positions or len(self.positions) >= self.max_positions:
                continue
            
            df = data[symbol]
            if date not in df.index:
                continue
            
            price = df.loc[date, 'close']
            quantity = self._calculate_position_size(signal, price)
            
            if quantity > 0:
                adjusted_price = price * (1 + self.slippage * (1 if signal > 0 else -1))
                
                self.positions[symbol] = {
                    'entry_date': date,
                    'entry_price': adjusted_price,
                    'quantity': quantity,
                    'side': 'buy' if signal > 0 else 'sell',
                    'entry_signal': signal,
                    'max_price': adjusted_price,
                    'min_price': adjusted_price
                }
                
                if signal > 0:
                    self.capital -= quantity * adjusted_price * (1 + self.commission)
                else:
                    self.capital += quantity * adjusted_price * (1 - self.commission)
    
    def _close_positions(self, date: datetime, data: Dict[str, pd.DataFrame],
                         stop_loss: Optional[float], take_profit: Optional[float],
                         trailing_stop: Optional[float], max_holding_days: Optional[int]):
        """Chiude le posizioni che hanno raggiunto target/stop."""
        to_close = []
        
        for symbol, position in self.positions.items():
            df = data[symbol]
            if date not in df.index:
                continue
            
            current_price = df.loc[date, 'close']
            
            # Aggiorna max/min
            if position['side'] == 'buy':
                position['max_price'] = max(position['max_price'], current_price)
                position['min_price'] = min(position['min_price'], current_price)
            else:
                position['max_price'] = max(position['max_price'], current_price)
                position['min_price'] = min(position['min_price'], current_price)
            
            # Check stop loss
            if stop_loss is not None:
                if position['side'] == 'buy':
                    stop_price = position['entry_price'] * (1 - stop_loss)
                    if current_price <= stop_price:
                        to_close.append((symbol, 'stop_loss'))
                        continue
                else:
                    stop_price = position['entry_price'] * (1 + stop_loss)
                    if current_price >= stop_price:
                        to_close.append((symbol, 'stop_loss'))
                        continue
            
            # Check take profit
            if take_profit is not None:
                if position['side'] == 'buy':
                    target_price = position['entry_price'] * (1 + take_profit)
                    if current_price >= target_price:
                        to_close.append((symbol, 'take_profit'))
                        continue
                else:
                    target_price = position['entry_price'] * (1 - take_profit)
                    if current_price <= target_price:
                        to_close.append((symbol, 'take_profit'))
                        continue
            
            # Check trailing stop
            if trailing_stop is not None:
                if position['side'] == 'buy':
                    trailing_price = position['max_price'] * (1 - trailing_stop)
                    if current_price <= trailing_price:
                        to_close.append((symbol, 'trailing_stop'))
                        continue
                else:
                    trailing_price = position['min_price'] * (1 + trailing_stop)
                    if current_price >= trailing_price:
                        to_close.append((symbol, 'trailing_stop'))
                        continue
            
            # Check max holding days
            if max_holding_days is not None:
                holding_days = (date - position['entry_date']).days
                if holding_days >= max_holding_days:
                    to_close.append((symbol, 'max_holding'))
                    continue
        
        for symbol, reason in to_close:
            self._close_position(symbol, date, data, reason)
    
    def _close_position(self, symbol: str, date: datetime,
                        data: Dict[str, pd.DataFrame], reason: str):
        """Chiude una singola posizione."""
        position = self.positions[symbol]
        df = data[symbol]
        current_price = df.loc[date, 'close']
        
        if position['side'] == 'buy':
            pnl = (current_price - position['entry_price']) * position['quantity']
            pnl_percent = (current_price / position['entry_price'] - 1) * 100
            exit_price = current_price * (1 - self.slippage)
        else:
            pnl = (position['entry_price'] - current_price) * position['quantity']
            pnl_percent = (position['entry_price'] / current_price - 1) * 100
            exit_price = current_price * (1 + self.slippage)
        
        pnl -= abs(pnl) * self.commission
        
        trade = Trade(
            symbol=symbol,
            entry_date=position['entry_date'],
            exit_date=date,
            entry_price=position['entry_price'],
            exit_price=exit_price,
            quantity=position['quantity'],
            side=position['side'],
            pnl=pnl,
            pnl_percent=pnl_percent,
            holding_days=(date - position['entry_date']).days,
            max_favorable_excursion=position['max_price'] - position['entry_price'],
            max_adverse_excursion=position['min_price'] - position['entry_price'],
            exit_reason=reason
        )
        
        self.trades.append(trade)
        del self.positions[symbol]
        
        if trade.side == 'buy':
            self.capital += position['quantity'] * exit_price * (1 - self.commission)
        else:
            self.capital -= position['quantity'] * exit_price * (1 + self.commission)
    
    def _calculate_position_size(self, signal: float, price: float) -> int:
        """Calcola la dimensione della posizione."""
        if self.position_sizing == 'kelly':
            win_rate = self._calculate_win_rate()
            avg_win = self._calculate_avg_win()
            avg_loss = self._calculate_avg_loss()
            
            if avg_loss > 0:
                kelly = (win_rate * (avg_win / avg_loss) - (1 - win_rate)) / (avg_win / avg_loss)
                kelly = max(0, min(kelly, 0.25))
                position_value = self.capital * kelly * abs(signal)
            else:
                position_value = self.capital * 0.05 * abs(signal)
        
        elif self.position_sizing == 'volatility':
            position_value = self.capital * 0.05 * abs(signal)
        
        else:
            position_value = self.capital * self.risk_per_trade * abs(signal)
        
        max_position_value = self.capital * 0.20
        position_value = min(position_value, max_position_value)
        
        return int(position_value / price) if price > 0 else 0
    
    def _update_equity(self, date: datetime, data: Dict[str, pd.DataFrame]) -> float:
        """Aggiorna il valore del portafoglio."""
        total_value = self.capital
        
        for symbol, position in self.positions.items():
            df = data[symbol]
            if date in df.index:
                price = df.loc[date, 'close']
                total_value += price * position['quantity']
        
        return total_value
    
    def _calculate_win_rate(self) -> float:
        """Calcola il win rate storico."""
        if not self.trades:
            return 0.5
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades) if self.trades else 0
    
    def _calculate_avg_win(self) -> float:
        """Calcola il profitto medio."""
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        return np.mean(wins) if wins else 0
    
    def _calculate_avg_loss(self) -> float:
        """Calcola la perdita media."""
        losses = [t.pnl for t in self.trades if t.pnl < 0]
        return abs(np.mean(losses)) if losses else 0
    
    def _calculate_metrics(self) -> BacktestResult:
        """Calcola tutte le metriche di performance."""
        if not self.trades:
            return BacktestResult(
                total_return=0, annual_return=0, volatility=0,
                sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
                max_drawdown=0, max_drawdown_duration=0,
                total_trades=0, win_rate=0, avg_win=0, avg_loss=0,
                profit_factor=0, expectancy=0, avg_holding_days=0,
                equity_curve=pd.Series(), drawdown_curve=pd.Series(),
                trades=[], monthly_returns=pd.DataFrame(),
                yearly_returns=pd.DataFrame(),
                var_95=0, cvar_95=0
            )
        
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        total_return = (equity[-1] / equity[0] - 1)
        annual_return = (1 + total_return) ** (252 / len(equity)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.03) / volatility if volatility > 0 else 0
        
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        max_dd_idx = drawdown.argmin()
        max_dd_duration = 0
        for i in range(max_dd_idx, -1, -1):
            if drawdown[i] == 0:
                max_dd_duration = max_dd_idx - i
                break
        
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - 0.03) / downside_deviation if downside_deviation > 0 else 0
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        total_trades = len(self.trades)
        pnls = np.array([t.pnl for t in self.trades])
        win_rate = (pnls > 0).mean()
        avg_win = pnls[pnls > 0].mean() if (pnls > 0).any() else 0
        avg_loss = abs(pnls[pnls < 0].mean()) if (pnls < 0).any() else 0
        profit_factor = abs(pnls[pnls > 0].sum() / pnls[pnls < 0].sum()) if (pnls < 0).any() else np.inf
        expectancy = pnls.mean()
        avg_holding_days = np.mean([t.holding_days for t in self.trades])
        
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_holding_days=avg_holding_days,
            equity_curve=pd.Series(equity),
            drawdown_curve=pd.Series(drawdown),
            trades=self.trades,
            monthly_returns=pd.DataFrame(),
            yearly_returns=pd.DataFrame(),
            var_95=var_95,
            cvar_95=cvar_95
        )