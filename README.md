cat > README.md << 'EOF'
# 🤖 Trading Bot - AI-Powered Trading System

Bot di trading automatico che utilizza un ensemble di algoritmi di intelligenza artificiale per generare segnali di trading su azioni.

## 📊 Algoritmi Integrati

### 1. Clenow Momentum Score
- Regressione esponenziale su 125 giorni
- Penalizzazione per volatilità con R²
- Ranking sistematico dei titoli

### 2. Raschke Patterns
- Oscillatore 3/10
- Divergenze prezzo/oscillatore
- First Cross Buy/Sell
- Anti Pattern

### 3. Monte Carlo Risk (Brandimarte)
- Simulazioni Monte Carlo per VaR
- Stima volatilità e Sharpe ratio
- Analisi del rischio di portafoglio

## 🧠 Machine Learning

- **XGBoost**: Modello robusto con feature importance
- **Neural Network**: Pattern complessi e non lineari
- **Ensemble**: Combinazione pesata dei modelli

## 🏗️ Architettura
