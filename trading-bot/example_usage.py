# example_usage.py
import pandas as pd
import yfinance as yf
from signals.clenow import ClenowMomentum
from signals.raschke import RaschkePatterns
from signals.brandimarte import MonteCarloRisk
from signals.ensemble import SignalEnsemble

# Scarica i dati
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
data = {}
for symbol in symbols:
    ticker = yf.Ticker(symbol)
    data[symbol] = ticker.history(period='2y')['Close']

# 1. Clenow Momentum
clenow = ClenowMomentum()
scores = clenow.calculate_ranked(data)
print("Clenow Ranking:")
print(scores.head())
print()

# 2. Raschke Patterns
raschke = RaschkePatterns()
for symbol, prices in data.items():
    signal = raschke.calculate(prices)
    components = raschke.calculate_components(prices)
    print(f"{symbol}: Raschke Signal = {signal:.3f}")
    print(f"  Components: {components}")
print()

# 3. Monte Carlo Risk
risk = MonteCarloRisk()
for symbol, prices in data.items():
    metrics = risk.calculate_metrics(prices)
    if metrics:
        print(f"{symbol}:")
        print(f"  Volatility: {metrics.volatility:.2%}")
        print(f"  VaR 95%: {metrics.var_95:.2%}")
        print(f"  Sharpe: {metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
print()

# 4. Ensemble Signal
ensemble = SignalEnsemble()
results = ensemble.calculate_batch(data)

print("Ensemble Results (Gradimento):")
for symbol, result in results.items():
    print(f"{symbol}: {result.gradimento:.3f} ({result.confidence:.2%})")
    for line in result.interpretation:
        print(f"  {line}")

# 5. Ranking
print("\nRanking:")
ranking = ensemble.rank_symbols(data)
print(ranking[['symbol', 'gradimento', 'confidence', 'rank']].to_string())

# 6. Trading Signals
print("\nTrading Signals:")
for symbol, result in results.items():
    signal = ensemble.get_trading_signal(result)
    print(f"{symbol}: {signal['action']} (size: {signal['position_size']:.1%})")