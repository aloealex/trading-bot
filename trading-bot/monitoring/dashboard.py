# monitoring/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

class TradingDashboard:
    """
    Dashboard di monitoraggio in tempo reale per il bot di trading.
    """
    
    def __init__(self, bot):
        self.bot = bot
        self.update_interval = 5  # secondi
        self.status_history = []
        self.price_history = {}
    
    def run(self):
        """
        Avvia la dashboard Streamlit.
        """
        st.set_page_config(
            page_title="Trading Bot Dashboard",
            page_icon="📊",
            layout="wide"
        )
        
        st.title("🤖 Trading Bot - Dashboard di Monitoraggio")
        
        # Sidebar con controlli
        with st.sidebar:
            st.header("Controlli")
            
            if st.button("🔄 Aggiorna"):
                self._update_data()
            
            if st.button("🛑 Arresta Bot"):
                self.bot.is_running = False
                st.success("Bot arrestato")
            
            st.header("Info Bot")
            status = self.bot.get_status()
            st.metric("Capitale", f"${status['capital']:,.2f}")
            st.metric("Posizioni Aperte", len(status['positions']))
            st.metric("PnL Giornaliero", f"${status['daily_pnl']:,.2f}")
            st.metric("Total Trades", status['total_trades'])
        
        # Layout principale
        col1, col2 = st.columns(2)
        
        # Colonna 1: Segnali Correnti
        with col1:
            st.subheader("📈 Segnali Correnti")
            signals_df = self._get_signals_df()
            
            if not signals_df.empty:
                # Colora in base al gradimento
                def color_signal(val):
                    if val > 0.5:
                        return 'background-color: #00ff00; color: black'
                    elif val > 0.2:
                        return 'background-color: #90ee90; color: black'
                    elif val < -0.5:
                        return 'background-color: #ff0000; color: white'
                    elif val < -0.2:
                        return 'background-color: #ff6b6b; color: black'
                    else:
                        return 'background-color: #ffff00; color: black'
                
                styled_df = signals_df.style.applymap(color_signal, subset=['Gradimento'])
                st.dataframe(styled_df, use_container_width=True)
            
            # Posizioni Aperte
            st.subheader("📊 Posizioni Aperte")
            positions_df = self._get_positions_df()
            
            if not positions_df.empty:
                st.dataframe(positions_df, use_container_width=True)
            else:
                st.info("Nessuna posizione aperta")
        
        # Colonna 2: Grafici
        with col2:
            st.subheader("📉 Equity Curve")
            fig = self._create_equity_chart()
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📊 Performance Settimanale")
            weekly_fig = self._create_weekly_performance()
            st.plotly_chart(weekly_fig, use_container_width=True)
        
        # Sezione in fondo: Trade History
        st.subheader("📋 Cronologia Trades")
        trades_df = self._get_trades_df()
        if not trades_df.empty:
            st.dataframe(trades_df, use_container_width=True)
        
        # Auto-update
        if st.checkbox("Auto-update", value=True):
            time.sleep(self.update_interval)
            st.rerun()
    
    def _get_signals_df(self) -> pd.DataFrame:
        """Ottiene i segnali correnti come DataFrame"""
        evaluations = self.bot.evaluate_all_symbols()
        
        data = []
        for eval_data in evaluations:
            data.append({
                'Symbol': eval_data.get('symbol', ''),
                'Gradimento': eval_data.get('gradimento', 0),
                'Prezzo': eval_data.get('price', 0),
                'Timestamp': eval_data.get('timestamp', datetime.now())
            })
        
        return pd.DataFrame(data).sort_values('Gradimento', ascending=False)
    
    def _get_positions_df(self) -> pd.DataFrame:
        """Ottiene le posizioni aperte come DataFrame"""
        data = []
        for symbol, pos in self.bot.positions.items():
            current_price = self.bot.broker.get_current_price(symbol)
            pnl = (current_price - pos['entry_price']) * pos['quantity']
            pnl_percent = (current_price / pos['entry_price'] - 1) * 100
            
            data.append({
                'Symbol': symbol,
                'Quantità': pos['quantity'],
                'Entry Price': pos['entry_price'],
                'Current Price': current_price,
                'PnL': f"${pnl:,.2f} ({pnl_percent:.1f}%)",
                'Entry Date': pos['entry_date']
            })
        
        return pd.DataFrame(data)
    
    def _get_trades_df(self) -> pd.DataFrame:
        """Ottiene la cronologia dei trades"""
        trades = self.bot.order_manager.closed_orders
        data = []
        for trade in trades:
            data.append({
                'Symbol': trade.symbol,
                'Side': trade.side.value,
                'Quantity': trade.quantity,
                'Price': trade.filled_price,
                'Date': trade.filled_at,
                'PnL': trade.pnl if hasattr(trade, 'pnl') else 0
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values('Date', ascending=False)
        
        return df
    
    def _create_equity_chart(self) -> go.Figure:
        """Crea il grafico dell'equity curve"""
        # Simula l'equity curve (da sostituire con dati reali)
        dates = pd.date_range(end=datetime.now(), periods=100)
        equity = 100000 + np.cumsum(np.random.randn(100) * 1000)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity,
            mode='lines',
            name='Equity',
            line=dict(color='#00cc96', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 204, 150, 0.1)'
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title='Data',
            yaxis_title='Equity ($)',
            yaxis_tickformat='$,.0f'
        )
        
        return fig
    
    def _create_weekly_performance(self) -> go.Figure:
        """Crea il grafico delle performance settimanali"""
        # Dati simulati
        weeks = ['Settimana 1', 'Settimana 2', 'Settimana 3', 'Settimana 4']
        returns = np.random.randn(4) * 0.02
        
        colors = ['#00cc96' if r > 0 else '#ff6b6b' for r in returns]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=weeks,
            y=returns * 100,
            marker_color=colors,
            text=[f"{r*100:.1f}%" for r in returns],
            textposition='auto'
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title='Settimana',
            yaxis_title='Return (%)',
            yaxis_tickformat='.1f%'
        )
        
        return fig

def run_dashboard(bot):
    """Avvia la dashboard per un bot specifico"""
    dashboard = TradingDashboard(bot)
    dashboard.run()