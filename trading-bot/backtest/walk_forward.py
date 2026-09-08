# backtest/walk_forward.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from .engine import BacktestEngine, BacktestResult

class WalkForwardAnalyzer:
    """
    Analisi Walk-Forward per validare la robustezza del modello.
    
    La strategia viene testata su finestre temporali mobili per valutare
    la stabilità delle performance nel tempo.
    """
    
    def __init__(self,
                 train_window: int = 252,  # giorni di training
                 test_window: int = 63,    # giorni di testing
                 step_size: int = 21,      # passo di avanzamento
                 retrain_frequency: int = 1):  # ricostruisci il modello ogni N step
        
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.retrain_frequency = retrain_frequency
        self.results = []
    
    def run(self,
            data: Dict[str, pd.DataFrame],
            signal_generator: Any,  # Il tuo motore di segnali
            model_trainer: Any,    # Il trainer del modello
            **backtest_kwargs) -> pd.DataFrame:
        """
        Esegue l'analisi walk-forward.
        """
        dates = self._get_common_dates(data)
        results = []
        
        i = 0
        while i < len(dates) - self.train_window - self.test_window:
            # Split dei dati
            train_end = i + self.train_window
            test_end = train_end + self.test_window
            
            train_dates = dates[i:train_end]
            test_dates = dates[train_end:test_end]
            
            print(f"Window {i//self.step_size + 1}:")
            print(f"  Train: {train_dates[0].date()} -> {train_dates[-1].date()}")
            print(f"  Test: {test_dates[0].date()} -> {test_dates[-1].date()}")
            
            # 1. Addestra il modello sui dati di training
            train_data = {s: df.loc[train_dates] for s, df in data.items()}
            
            if i % (self.step_size * self.retrain_frequency) == 0:
                model = model_trainer.train(train_data)
            else:
                # Usa il modello precedente
                pass
            
            # 2. Genera segnali per il periodo di test
            signals = self._generate_signals(data, test_dates, signal_generator, model)
            
            # 3. Esegui il backtest sul periodo di test
            test_data = {s: df.loc[test_dates] for s, df in data.items()}
            
            engine = BacktestEngine(**backtest_kwargs)
            result = engine.run(test_data, signals)
            
            # 4. Registra i risultati
            results.append({
                'window_start': train_dates[0],
                'train_end': train_dates[-1],
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'profit_factor': result.profit_factor
            })
            
            i += self.step_size
        
        self.results = pd.DataFrame(results)
        
        return self.results
    
    def _get_common_dates(self, data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """Ottiene le date comuni a tutti i simboli"""
        all_dates = set(data[list(data.keys())[0]].index)
        for df in data.values():
            all_dates = all_dates.intersection(df.index)
        return sorted(all_dates)
    
    def _generate_signals(self, data: Dict[str, pd.DataFrame], 
                          dates: List[datetime],
                          signal_generator: Any,
                          model: Any) -> Dict[str, pd.Series]:
        """Genera i segnali per il periodo di test"""
        signals = {}
        
        for symbol, df in data.items():
            signal_series = pd.Series(index=dates, dtype=float)
            
            for date in dates:
                # Ottieni i prezzi fino a questa data
                prices = df.loc[:date, 'close']
                
                # Genera il segnale
                if model is not None:
                    # Usa il modello ML
                    features = signal_generator.extract_features(prices)
                    signal = model.predict(features)
                else:
                    # Usa il motore di segnali diretto
                    signal = signal_generator.calculate(prices)
                
                signal_series.loc[date] = signal
            
            signals[symbol] = signal_series
        
        return signals
    
    def analyze_results(self) -> Dict[str, float]:
        """
        Analizza i risultati del walk-forward.
        """
        if self.results is None or self.results.empty:
            return {}
        
        analysis = {
            'avg_annual_return': self.results['annual_return'].mean(),
            'std_annual_return': self.results['annual_return'].std(),
            'avg_sharpe': self.results['sharpe_ratio'].mean(),
            'std_sharpe': self.results['sharpe_ratio'].std(),
            'avg_drawdown': self.results['max_drawdown'].mean(),
            'avg_win_rate': self.results['win_rate'].mean(),
            'total_trades': self.results['total_trades'].sum(),
            'avg_profit_factor': self.results['profit_factor'].mean(),
            'positive_windows': (self.results['total_return'] > 0).sum(),
            'total_windows': len(self.results),
            'consistency': (self.results['total_return'] > 0).mean()
        }
        
        return analysis
    
    def plot_results(self):
        """Visualizza i risultati del walk-forward"""
        import matplotlib.pyplot as plt
        
        if self.results is None or self.results.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Annual Return per finestra
        axes[0, 0].bar(range(len(self.results)), self.results['annual_return'])
        axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 0].set_title('Annual Return per Window')
        axes[0, 0].set_xlabel('Window')
        axes[0, 0].set_ylabel('Annual Return')
        
        # 2. Sharpe Ratio per finestra
        axes[0, 1].bar(range(len(self.results)), self.results['sharpe_ratio'])
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0, 1].axhline(y=1, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
        axes[0, 1].set_title('Sharpe Ratio per Window')
        axes[0, 1].set_xlabel('Window')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        
        # 3. Max Drawdown per finestra
        axes[1, 0].bar(range(len(self.results)), self.results['max_drawdown'])
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].set_title('Max Drawdown per Window')
        axes[1, 0].set_xlabel('Window')
        axes[1, 0].set_ylabel('Max Drawdown')
        
        # 4. Win Rate per finestra
        axes[1, 1].bar(range(len(self.results)), self.results['win_rate'])
        axes[1, 1].axhline(y=0.5, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
        axes[1, 1].set_title('Win Rate per Window')
        axes[1, 1].set_xlabel('Window')
        axes[1, 1].set_ylabel('Win Rate')
        
        plt.tight_layout()
        plt.show()