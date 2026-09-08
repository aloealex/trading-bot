# run.py
#!/usr/bin/env python
"""
Script di avvio principale per il bot di trading.
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Carica le variabili d'ambiente
load_dotenv()

from deployment.deploy import TradingBotDeployment
from monitoring.dashboard import run_dashboard

def main():
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--config', '-c', 
                       default='config/production.json',
                       help='Percorso del file di configurazione')
    parser.add_argument('--mode', '-m',
                       choices=['trade', 'dashboard', 'backtest'],
                       default='trade',
                       help='Modalità di esecuzione')
    parser.add_argument('--log-level', '-l',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Livello di logging')
    parser.add_argument('--symbols', '-s',
                       nargs='+',
                       help='Lista di simboli da monitorare')
    
    args = parser.parse_args()
    
    # Crea il bot
    bot = TradingBotDeployment(
        config_path=args.config,
        log_level=args.log_level
    )
    
    # Sovrascrivi i simboli se specificati
    if args.symbols:
        bot.symbols = args.symbols
    
    if args.mode == 'trade':
        # Avvia il trading
        bot.run()
    elif args.mode == 'dashboard':
        # Avvia la dashboard
        if bot.initialize():
            run_dashboard(bot)
    elif args.mode == 'backtest':
        # Esegui il backtesting
        from backtest.walk_forward import WalkForwardAnalyzer
        from backtest.engine import BacktestEngine
        
        analyzer = WalkForwardAnalyzer()
        # (implementazione del backtest...)
        print("Modalità backtest (da implementare)")

if __name__ == '__main__':
    main()