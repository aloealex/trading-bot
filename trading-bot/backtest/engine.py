# backtest/engine.py
"""
Motore di backtest giornaliero.

Regole (pensate per evitare risultati gonfiati):
- il segnale della data t è calcolato con i prezzi fino alla chiusura di t;
- l'ordine viene eseguito all'APERTURA di t+1 (o alla chiusura di t+1 se manca l'open):
  niente esecuzione "al prezzo che ha generato il segnale";
- cassa (cash) ed equity sono tenute separate;
- stop loss / take profit / trailing / max holding sono controllati sulla chiusura e
  la posizione esce all'apertura successiva, come farebbe il bot live;
- commissioni e slippage applicati a ogni esecuzione.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Rappresenta una singola operazione chiusa."""
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
    monthly_returns: pd.Series
    yearly_returns: pd.Series
    var_95: float
    cvar_95: float
    exposure: float = 0.0

    def summary(self) -> Dict[str, float]:
        return {
            'Rendimento totale': self.total_return,
            'Rendimento annuo': self.annual_return,
            'Volatilità annua': self.volatility,
            'Sharpe': self.sharpe_ratio,
            'Sortino': self.sortino_ratio,
            'Calmar': self.calmar_ratio,
            'Max drawdown': self.max_drawdown,
            'Durata max DD (barre)': self.max_drawdown_duration,
            'Operazioni': self.total_trades,
            'Win rate': self.win_rate,
            'Profit factor': self.profit_factor,
            'Expectancy ($)': self.expectancy,
            'Giorni medi in posizione': self.avg_holding_days,
            'Esposizione media': self.exposure,
        }


class BacktestEngine:
    """Motore di backtesting."""

    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 max_positions: int = 10,
                 position_sizing: str = 'fixed',
                 risk_per_trade: float = 0.10,
                 max_position_size: float = 0.20,
                 entry_threshold: float = 0.3,
                 exit_threshold: float = -0.3,
                 allow_short: bool = False,
                 risk_free_rate: float = 0.03,
                 min_trades_for_kelly: int = 20):
        """
        position_sizing:
            'fixed'  -> risk_per_trade * equity * |segnale|
            'kelly'  -> Kelly frazionato (1/4) stimato dai trade passati; finché non ci sono
                        almeno `min_trades_for_kelly` operazioni usa 'fixed'
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_positions = max_positions
        self.position_sizing = position_sizing
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.allow_short = allow_short
        self.risk_free_rate = risk_free_rate
        self.min_trades_for_kelly = min_trades_for_kelly
        self._reset()

    def _reset(self):
        self.cash = float(self.initial_capital)
        self.positions: Dict[str, dict] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.equity_dates: List[datetime] = []
        self._exposure: List[float] = []
        self.results = None

    # ------------------------------------------------------------------ #
    def run(self,
            data: Dict[str, pd.DataFrame],
            signals: Dict[str, pd.Series],
            stop_loss: Optional[float] = None,
            take_profit: Optional[float] = None,
            trailing_stop: Optional[float] = None,
            max_holding_days: Optional[int] = None) -> BacktestResult:
        """Esegue il backtest completo."""
        self._reset()
        dates = sorted(set().union(*[df.index for df in data.values()]))
        signals = {s: sig.dropna() for s, sig in signals.items()}

        pending_exits: Dict[str, str] = {}
        pending_entries: Dict[str, float] = {}

        for date in dates:
            # 1. Esegui all'apertura gli ordini decisi alla chiusura precedente
            for symbol, reason in list(pending_exits.items()):
                if symbol in self.positions and self._has_bar(data, symbol, date):
                    self._close_position(symbol, date, self._exec_price(data[symbol], date), reason)
                    pending_exits.pop(symbol)
            for symbol, signal in sorted(pending_entries.items(), key=lambda kv: -abs(kv[1])):
                if len(self.positions) >= self.max_positions:
                    break
                if symbol not in self.positions and self._has_bar(data, symbol, date):
                    self._open_position(symbol, date, self._exec_price(data[symbol], date), signal, data)
            pending_entries = {}

            # 2. Alla chiusura: aggiorna le posizioni e decidi le uscite
            for symbol, pos in self.positions.items():
                if not self._has_bar(data, symbol, date) or symbol in pending_exits:
                    continue
                close = float(data[symbol].at[date, 'close'])
                pos['max_price'] = max(pos['max_price'], close)
                pos['min_price'] = min(pos['min_price'], close)
                reason = self._exit_reason(pos, close, date, stop_loss, take_profit,
                                           trailing_stop, max_holding_days)
                if reason is None:
                    sig = self._signal_at(signals.get(symbol), date)
                    if sig is not None:
                        if pos['side'] == 'buy' and sig < self.exit_threshold:
                            reason = 'signal'
                        elif pos['side'] == 'sell' and sig > -self.exit_threshold:
                            reason = 'signal'
                if reason:
                    pending_exits[symbol] = reason

            # 3. Alla chiusura: nuovi ingressi per il giorno dopo
            for symbol, series in signals.items():
                if symbol in self.positions or symbol not in data:
                    continue
                sig = self._signal_at(series, date)
                if sig is None:
                    continue
                if sig > self.entry_threshold or (self.allow_short and sig < -self.entry_threshold):
                    pending_entries[symbol] = sig

            # 4. Equity a fine giornata
            equity, gross = self._mark_to_market(data, date)
            self.equity_curve.append(equity)
            self.equity_dates.append(date)
            self._exposure.append(gross / equity if equity > 0 else 0.0)

        # Chiudi ciò che resta all'ultima chiusura disponibile
        if dates:
            for symbol in list(self.positions):
                df = data[symbol]
                last = df.index[df.index <= dates[-1]][-1]
                self._close_position(symbol, last, float(df.at[last, 'close']), 'end_of_test')
            if self.equity_curve:
                self.equity_curve[-1] = self.cash

        self.results = self._calculate_metrics()
        return self.results

    # ------------------------------------------------------------------ #
    @staticmethod
    def _has_bar(data, symbol, date) -> bool:
        return symbol in data and date in data[symbol].index

    @staticmethod
    def _exec_price(df: pd.DataFrame, date) -> float:
        if 'open' in df.columns and pd.notna(df.at[date, 'open']) and df.at[date, 'open'] > 0:
            return float(df.at[date, 'open'])
        return float(df.at[date, 'close'])

    @staticmethod
    def _signal_at(series: Optional[pd.Series], date) -> Optional[float]:
        if series is None or date not in series.index:
            return None
        value = series.loc[date]
        return float(value) if pd.notna(value) else None

    def _exit_reason(self, pos, close, date, stop_loss, take_profit, trailing_stop, max_holding_days):
        entry = pos['entry_price']
        long = pos['side'] == 'buy'
        if stop_loss is not None:
            if (long and close <= entry * (1 - stop_loss)) or (not long and close >= entry * (1 + stop_loss)):
                return 'stop_loss'
        if take_profit is not None:
            if (long and close >= entry * (1 + take_profit)) or (not long and close <= entry * (1 - take_profit)):
                return 'take_profit'
        if trailing_stop is not None:
            if long and close <= pos['max_price'] * (1 - trailing_stop):
                return 'trailing_stop'
            if not long and close >= pos['min_price'] * (1 + trailing_stop):
                return 'trailing_stop'
        if max_holding_days is not None and (date - pos['entry_date']).days >= max_holding_days:
            return 'max_holding'
        return None

    def _current_equity(self) -> float:
        value = self.cash
        for pos in self.positions.values():
            mark = pos.get('last_price', pos['entry_price'])
            value += mark * pos['quantity'] if pos['side'] == 'buy' else -mark * pos['quantity']
        return value

    def _position_value(self, signal: float, price: float) -> float:
        equity = self._current_equity()
        fraction = self.risk_per_trade
        if self.position_sizing == 'kelly' and len(self.trades) >= self.min_trades_for_kelly:
            pnls = np.array([t.pnl_percent for t in self.trades])
            wins, losses = pnls[pnls > 0], pnls[pnls < 0]
            if len(wins) and len(losses):
                b = wins.mean() / abs(losses.mean())
                p = len(wins) / len(pnls)
                kelly = (p * b - (1 - p)) / b
                # Kelly frazionato; se l'edge stimato è nullo si torna a una quota minima
                # invece di smettere di operare per sempre.
                fraction = float(np.clip(kelly * 0.25, self.risk_per_trade * 0.25, self.max_position_size))
        value = equity * fraction * min(1.0, abs(signal))
        return min(value, equity * self.max_position_size)

    def _open_position(self, symbol, date, price, signal, data):
        side = 'buy' if signal > 0 else 'sell'
        fill = price * (1 + self.slippage) if side == 'buy' else price * (1 - self.slippage)
        quantity = int(self._position_value(signal, fill) / fill)
        if quantity <= 0:
            return
        fee = fill * quantity * self.commission
        if side == 'buy':
            if fill * quantity + fee > self.cash:
                quantity = int(self.cash / (fill * (1 + self.commission)))
                if quantity <= 0:
                    return
                fee = fill * quantity * self.commission
            self.cash -= fill * quantity + fee
        else:
            self.cash += fill * quantity - fee
        self.positions[symbol] = {
            'entry_date': date, 'entry_price': fill, 'quantity': quantity, 'side': side,
            'entry_signal': signal, 'entry_fee': fee,
            'max_price': fill, 'min_price': fill, 'last_price': fill,
        }

    def _close_position(self, symbol, date, price, reason):
        pos = self.positions.pop(symbol)
        qty = pos['quantity']
        if pos['side'] == 'buy':
            fill = price * (1 - self.slippage)
            fee = fill * qty * self.commission
            self.cash += fill * qty - fee
            gross = (fill - pos['entry_price']) * qty
        else:
            fill = price * (1 + self.slippage)
            fee = fill * qty * self.commission
            self.cash -= fill * qty + fee
            gross = (pos['entry_price'] - fill) * qty
        pnl = gross - fee - pos['entry_fee']
        cost_basis = pos['entry_price'] * qty
        self.trades.append(Trade(
            symbol=symbol,
            entry_date=pos['entry_date'],
            exit_date=date,
            entry_price=pos['entry_price'],
            exit_price=fill,
            quantity=qty,
            side=pos['side'],
            pnl=pnl,
            pnl_percent=pnl / cost_basis * 100 if cost_basis else 0.0,
            holding_days=(date - pos['entry_date']).days,
            max_favorable_excursion=(pos['max_price'] - pos['entry_price']) if pos['side'] == 'buy'
            else (pos['entry_price'] - pos['min_price']),
            max_adverse_excursion=(pos['min_price'] - pos['entry_price']) if pos['side'] == 'buy'
            else (pos['entry_price'] - pos['max_price']),
            exit_reason=reason,
        ))

    def _mark_to_market(self, data, date):
        equity, gross = self.cash, 0.0
        for symbol, pos in self.positions.items():
            if self._has_bar(data, symbol, date):
                pos['last_price'] = float(data[symbol].at[date, 'close'])
            value = pos['last_price'] * pos['quantity']
            gross += value
            equity += value if pos['side'] == 'buy' else -value
        return equity, gross

    # ------------------------------------------------------------------ #
    def _calculate_metrics(self) -> BacktestResult:
        equity = pd.Series(self.equity_curve, index=pd.DatetimeIndex(self.equity_dates), dtype=float)
        if equity.empty:
            equity = pd.Series([self.initial_capital], dtype=float)
        start_value = float(self.initial_capital)
        full = pd.concat([pd.Series([start_value]), equity.reset_index(drop=True)])
        returns = full.pct_change().dropna()

        total_return = equity.iloc[-1] / start_value - 1
        n = max(len(equity), 1)
        annual_return = (1 + total_return) ** (252 / n) - 1 if total_return > -1 else -1.0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
        excess = returns - self.risk_free_rate / 252
        sharpe = excess.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
        downside = returns[returns < 0]
        sortino = (excess.mean() * 252) / (downside.std() * np.sqrt(252)) if len(downside) > 1 and downside.std() > 0 else 0.0

        running_max = equity.cummax().clip(lower=start_value)
        drawdown = equity / running_max - 1
        max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
        # durata massima (in barre) passata sotto il massimo precedente
        underwater = (drawdown < 0).astype(int).values
        longest, run = 0, 0
        for u in underwater:
            run = run + 1 if u else 0
            longest = max(longest, run)
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

        pnls = np.array([t.pnl for t in self.trades])
        if len(pnls):
            win_rate = float((pnls > 0).mean())
            avg_win = float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0.0
            avg_loss = float(abs(pnls[pnls < 0].mean())) if (pnls < 0).any() else 0.0
            losses_sum = abs(pnls[pnls < 0].sum())
            profit_factor = float(pnls[pnls > 0].sum() / losses_sum) if losses_sum > 0 else float('inf')
            expectancy = float(pnls.mean())
            avg_holding = float(np.mean([t.holding_days for t in self.trades]))
        else:
            win_rate = avg_win = avg_loss = profit_factor = expectancy = avg_holding = 0.0

        if isinstance(equity.index, pd.DatetimeIndex) and len(equity) > 1:
            monthly = equity.resample('ME').last().pct_change().dropna()
            yearly = equity.resample('YE').last().pct_change().dropna()
        else:
            monthly = yearly = pd.Series(dtype=float)

        var_95 = float(np.percentile(returns, 5)) if len(returns) else 0.0
        tail = returns[returns <= var_95]
        cvar_95 = float(tail.mean()) if len(tail) else 0.0

        return BacktestResult(
            total_return=float(total_return),
            annual_return=float(annual_return),
            volatility=float(volatility),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar),
            max_drawdown=max_drawdown,
            max_drawdown_duration=int(longest),
            total_trades=len(self.trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_holding_days=avg_holding,
            equity_curve=equity,
            drawdown_curve=drawdown,
            trades=list(self.trades),
            monthly_returns=monthly,
            yearly_returns=yearly,
            var_95=var_95,
            cvar_95=cvar_95,
            exposure=float(np.mean(self._exposure)) if self._exposure else 0.0,
        )

    def trades_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([t.__dict__ for t in self.trades])
