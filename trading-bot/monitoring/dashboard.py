# monitoring/dashboard.py
"""
Dashboard Streamlit del bot.

Avvio:   python main.py --mode dashboard
   (oppure: streamlit run monitoring/dashboard.py)

Legge il file di stato che il bot scrive a ogni ciclo (state/bot_state.json):
mostra solo dati REALI. La versione precedente disegnava un'equity curve e dei
rendimenti settimanali generati con np.random.
"""
import json
import os
import time
from datetime import datetime

import pandas as pd

STATE_FILE = os.environ.get('BOT_STATE_FILE', os.path.join('state', 'bot_state.json'))


def load_state(path: str = STATE_FILE):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _color_gradimento(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ''
    if v > 0.3:
        return 'background-color: #c8e6c9'
    if v < -0.3:
        return 'background-color: #ffcdd2'
    return ''


def render():
    import streamlit as st
    import plotly.graph_objects as go

    st.set_page_config(page_title="Trading Bot", page_icon="📊", layout="wide")
    st.title("Trading Bot — monitoraggio")

    state = load_state()
    if state is None:
        st.warning(f"Nessuno stato trovato in `{STATE_FILE}`. Avvia prima il bot: `python main.py --mode trade`.")
        return

    age = (datetime.now() - datetime.fromisoformat(state['updated_at'])).total_seconds()
    mode = 'SIMULATO' if state.get('broker') == 'sim' else ('PAPER' if state.get('paper') else 'REALE')
    st.caption(f"Broker: {state.get('broker')} ({mode}) — aggiornato il {state['updated_at']} "
               f"({int(age // 60)} min fa)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity", f"{state.get('equity', 0):,.2f}")
    c2.metric("Liquidità", f"{state.get('cash', 0):,.2f}")
    c3.metric("Posizioni aperte", len(state.get('positions', [])))
    c4.metric("Ordini eseguiti", state.get('order_summary', {}).get('filled_orders', 0))

    left, right = st.columns(2)
    with left:
        st.subheader("Segnali dell'ultimo ciclo")
        ev = pd.DataFrame(state.get('evaluations', []))
        if not ev.empty:
            cols = [c for c in ['symbol', 'gradimento', 'confidence', 'price', 'volatility', 'source', 'error']
                    if c in ev.columns]
            st.dataframe(ev[cols].style.map(_color_gradimento, subset=['gradimento'])
                         .format({'gradimento': '{:+.3f}', 'confidence': '{:.0%}',
                                  'price': '{:,.2f}', 'volatility': '{:.1%}'}, na_rep=''),
                         hide_index=True)
        st.subheader("Posizioni aperte")
        pos = pd.DataFrame(state.get('positions', []))
        if pos.empty:
            st.info("Nessuna posizione aperta")
        else:
            st.dataframe(pos, hide_index=True)

    with right:
        st.subheader("Valore del portafoglio")
        hist = pd.DataFrame(state.get('equity_history', []))
        if len(hist) > 1:
            hist['timestamp'] = pd.to_datetime(hist['timestamp'])
            fig = go.Figure(go.Scatter(x=hist['timestamp'], y=hist['total_value'], mode='lines'))
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), yaxis_title='Valore')
            st.plotly_chart(fig)
        else:
            st.info("Storico ancora insufficiente: compare dopo qualche ciclo del bot")

    st.subheader("Operazioni recenti")
    trades = pd.DataFrame(state.get('trades', []))
    if trades.empty:
        st.info("Nessuna operazione ancora")
    else:
        st.dataframe(trades.iloc[::-1], hide_index=True)

    if st.sidebar.checkbox("Aggiornamento automatico (30 s)", value=False):
        time.sleep(30)
        st.rerun()


if __name__ == '__main__':
    render()
