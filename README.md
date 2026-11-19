# 🤖 Trading Bot

Bot di trading azionario (solo posizioni long) che combina tre approcci quantitativi in un unico punteggio, il **gradimento** (da -1 a +1). In alternativa il punteggio può venire da un modello di machine learning.

| Componente | Cosa fa |
|---|---|
| **Clenow Momentum** | Regressione esponenziale su 125 giorni: pendenza annualizzata × R² |
| **Raschke 3/10** | Oscillatore 3/10 con slow line a 16: First Cross, divergenze e pattern Anti |
| **Monte Carlo (Brandimarte)** | Simulazioni bootstrap per volatilità, VaR/CVaR, probabilità di perdita, insieme allo Sharpe storico |
| **Mean reversion / RSI** | Efficiency Ratio di Kaufman e RSI di Wilder come correttivi |
| **ML (opzionale)** | XGBoost, rete neurale (PyTorch) o ensemble, che stimano il rendimento a 5 giorni |

Broker supportati: **Alpaca** (`alpaca-py`), **Interactive Brokers** (`ib_insync`, con TWS o IB Gateway) e un **broker simulato** per fare prove senza conto.

> ⚠️ Software sperimentale. Nessun risultato passato o simulato garantisce risultati futuri. Prima di usare denaro reale, fallo girare a lungo su un conto *paper*.

---

## Struttura

```
trading-bot/
├── main.py                 # punto di ingresso unico (tutte le modalità)
├── run.py                  # alias di main.py
├── example_usage.py        # esempio d'uso dei segnali
├── config/
│   ├── production.json     # configurazione principale
│   ├── settings.py         # dataclass delle impostazioni + caricamento JSON/.env
│   └── symbols.py          # watchlist
├── signals/                # Clenow, Raschke, Monte Carlo, ensemble
├── models/                 # feature, addestramento, predizione, XGBoost/NN/ensemble
├── risk/                   # dimensionamento posizioni, stop loss, rischio di portafoglio
├── execution/              # broker (Alpaca, IBKR, simulato), ordini, portafoglio
├── backtest/               # motore di backtest, metriche, walk-forward
├── data/                   # dati di mercato (yfinance, CSV)
├── monitoring/             # log, notifiche Telegram/email, dashboard Streamlit
└── tests/                  # test automatici (pytest)
```

## Installazione

Serve Python 3.10 o superiore.

```bash
cd trading-bot
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# facoltativo, solo per la rete neurale / ensemble:
pip install torch
```

## Configurazione

1. Copia `.env.example` in `.env` (nella cartella `trading-bot/`) e inserisci le chiavi API di Alpaca. Il file `.env` è escluso da git.
2. Modifica `config/production.json`. I campi principali:

| Campo | Significato |
|---|---|
| `broker` | `alpaca`, `ib` o `sim` |
| `broker_config.paper` | `true` = conto paper Alpaca. Mettilo a `false` **solo** per il conto reale |
| `symbols` | watchlist |
| `buy_threshold` / `sell_threshold` | soglie di gradimento per entrare e uscire |
| `position_size` | quota di equity per posizione, moltiplicata per il gradimento (0,10 × 0,5 = 5%) |
| `max_position_size`, `max_positions` | limiti per titolo e numero di posizioni |
| `stop_loss`, `trailing_stop`, `take_profit`, `max_holding_days` | regole di uscita |
| `use_ml`, `model_type`, `model_path` | uso del modello ML al posto dei segnali diretti |
| `ensemble_weights` | pesi di momentum, raschke, risk e mean_reversion |

## Uso

Tutti i comandi vanno lanciati dalla cartella `trading-bot/`.

```bash
# Classifica dei segnali di oggi (nessun ordine)
python main.py --mode signals

# Backtest dei segnali su 5 anni (dati yfinance), con salvataggio di equity e trade
python main.py --mode backtest --years 5 -o backtest_output

# Addestramento del modello ML e backtest sul periodo di test
python main.py --mode train --years 8
python main.py --mode backtest --ml

# Validazione walk-forward (riaddestra il modello a ogni finestra)
python main.py --mode walkforward --ml --years 8

# Bot con broker simulato (nessun ordine reale), un solo ciclo
python main.py --mode trade --broker sim --cycles 1

# Bot sul conto paper Alpaca
python main.py --mode trade

# Dashboard (legge lo stato che il bot salva a ogni ciclo)
python main.py --mode dashboard
```

Al posto di yfinance si possono usare dati propri con `--csv-dir CARTELLA`: un file per simbolo (`AAPL.csv`, …) con colonne `date, open, high, low, close, volume`.

### Percorso consigliato

1. `--mode backtest` e `--mode walkforward`: verifica che la strategia abbia un vantaggio **fuori campione**. Confrontala con il riferimento *compra e tieni* stampato a fine backtest.
2. `--mode trade --broker sim` e poi il conto paper Alpaca per qualche settimana.
3. Il conto reale solo dopo, con capitale ridotto.

## Come funziona il bot live

A ogni ciclo (`update_interval`, 5 minuti di default) il bot:

1. scarica lo storico giornaliero e il prezzo corrente di ogni simbolo e calcola il gradimento;
2. se il mercato è aperto:
   - chiude le posizioni che hanno toccato stop loss, trailing stop, take profit o durata massima, o il cui gradimento è sceso sotto `sell_threshold`;
   - apre nuove posizioni sui titoli con gradimento sopra `buy_threshold`, finché ci sono slot e liquidità;
3. controlla i limiti di rischio del portafoglio e invia gli eventuali avvisi;
4. salva lo stato in `state/bot_state.json` per la dashboard.

Le posizioni già aperte sul conto all'avvio vengono prese in carico e messe sotto stop.

## Backtest: regole

- Il segnale della data *t* usa solo i prezzi fino alla chiusura di *t* e viene eseguito all'**apertura di t+1**.
- Commissioni e slippage sono applicati a ogni esecuzione.
- Gli stop sono controllati sulle chiusure giornaliere.
- Il backtest con `--ml` è significativo solo sul periodo di test del modello (indicato in `meta.json`). Per una stima onesta usa `--mode walkforward --ml`.

## Test

```bash
cd trading-bot
python -m pytest -q
```

I test usano dati sintetici e il broker simulato: non serve connessione né un conto.
