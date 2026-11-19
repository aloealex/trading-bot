# example_usage.py
"""
Esempio d'uso dei segnali, senza broker né ordini.
    python example_usage.py                 # scarica i dati da yfinance
    python example_usage.py CARTELLA_CSV     # usa i tuoi CSV (uno per simbolo)
"""
import sys

from data.loaders import load_prices
from signals.clenow import ClenowMomentum
from signals.raschke import RaschkePatterns
from signals.brandimarte import MonteCarloRisk
from signals.ensemble import SignalEnsemble

symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
csv_dir = sys.argv[1] if len(sys.argv) > 1 else None
data = {s: df['close'] for s, df in load_prices(symbols, csv_dir=csv_dir, years=2).items()}
if not data:
    sys.exit("Nessun dato caricato")

# 1. Clenow Momentum
print("Clenow ranking:")
print(ClenowMomentum().calculate_ranked(data).round(2).to_string(), "\n")

# 2. Raschke Patterns
raschke = RaschkePatterns()
for symbol, prices in data.items():
    print(f"{symbol}: Raschke = {raschke.calculate(prices):+.3f}  ->  "
          f"{'; '.join(raschke.explain(prices)['interpretation'])}")
print()

# 3. Monte Carlo Risk
risk = MonteCarloRisk()
for symbol, prices in data.items():
    m = risk.calculate_metrics(prices)
    if m:
        print(f"{symbol}: vol {m.volatility:.1%}  VaR95 {m.var_95:.1%}  Sharpe {m.sharpe_ratio:.2f}  "
              f"MaxDD {m.max_drawdown:.1%}")
print()

# 4. Ensemble (gradimento)
ensemble = SignalEnsemble()
print(ensemble.rank_symbols(data)[['symbol', 'gradimento', 'confidence', 'momentum', 'raschke',
                                   'risk_score', 'rank']].round(3).to_string(index=False))
print()
for symbol, result in ensemble.calculate_batch(data).items():
    sig = ensemble.get_trading_signal(result)
    print(f"{symbol}: {sig['action']}")
    for line in result.interpretation:
        print(f"   {line}")
