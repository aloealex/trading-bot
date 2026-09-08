# dashboard.py
import streamlit as st
import pandas as pd
from signal_engine import SignalEngine
from ai_model import AIModel

st.title("IA Trading Advisor - Gradimenti Azioni")

# Lista delle azioni nel watchlist
watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

data = []
for symbol in watchlist:
    prices = get_prices(symbol)
    momentum = engine.momentum_score(prices)
    divergence = engine.divergence_signal(prices)
    volatility, var = engine.monte_carlo_risk(prices)
    
    features = [momentum, divergence, volatility, var]
    gradimento = model.predict(features)
    
    data.append({
        'Symbol': symbol,
        'Momentum': round(momentum, 2),
        'Divergence': round(divergence, 2),
        'Volatility': round(volatility, 2),
        'VaR': round(var, 2),
        'Gradimento': round(gradimento, 2),
        'Raccomandazione': 'COMPRA' if gradimento > 0.3 else 'VENDI' if gradimento < -0.3 else 'NEUTRO'
    })

df = pd.DataFrame(data)
st.dataframe(df.style.background_gradient(cmap='RdYlGn', subset=['Gradimento']))