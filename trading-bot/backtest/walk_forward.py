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
                 retrain_frequency: int = 1):  # riaddestra il modello ogni N finestre
        
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.retrain_frequency = retrain_frequency
        self.results = []
    
    def run(self,
            data: Dict[str, pd.DataFrame],
            signal_generator: Any = None,
            model_trainer: Any = None,
            verbose: bool = True,
            **backtest_kwargs) -> pd.DataFrame:
        """
        Esegue l'analisi walk-forward.

        signal_generator: un SignalEnsemble (segnali diretti), usato se model_trainer è None
        model_trainer:    un ModelTrainer; viene riaddestrato su ogni finestra di training
                          e usato per i segnali della finestra di test successiva
        """
        from models.predict import ModelPredictor

        dates = self._get_common_dates(data)
        results = []
        predictor = None

        i, window = 0, 0
        while i + self.train_window + self.test_window <= len(dates):
            train_dates = dates[i:i + self.train_window]
            test_dates = dates[i + self.train_window:i + self.train_window + self.test_window]
            if verbose:
                print(f"Finestra {window + 1}: train {train_dates[0].date()} -> {train_dates[-1].date()}, "
                      f"test {test_dates[0].date()} -> {test_dates[-1].date()}")

            if model_trainer is not None and (predictor is None or window % self.retrain_frequency == 0):
                train_data = {s: df.loc[train_dates[0]:train_dates[-1]] for s, df in data.items()}
                model_trainer.train(train_data)
                predictor = ModelPredictor()
                predictor.model, predictor.meta = model_trainer.model, model_trainer.meta
                predictor.features = model_trainer.meta['features']

            signals = {}
            for symbol, df in data.items():
                history = df.loc[:test_dates[-1]]  # solo passato e periodo di test: niente futuro
                if predictor is not None:
                    series = predictor.predict_series(history)
                else:
                    series = signal_generator.signal_series(
                        history['close'], start=history.index.get_loc(test_dates[0]) + 1)
                signals[symbol] = series.reindex(test_dates)

            test_data = {s: df.loc[test_dates[0]:test_dates[-1]] for s, df in data.items()}
            result = BacktestEngine(**backtest_kwargs).run(test_data, signals)

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
                'profit_factor': result.profit_factor,
            })
            i += self.step_size
            window += 1

        self.results = pd.DataFrame(results)
        return self.results

    @staticmethod
    def _get_common_dates(data: Dict[str, pd.DataFrame]) -> List[datetime]:
        """Ottiene le date comuni a tutti i simboli"""
        all_dates = set(data[list(data.keys())[0]].index)
        for df in data.values():
            all_dates = all_dates.intersection(df.index)
        return sorted(all_dates)
    
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