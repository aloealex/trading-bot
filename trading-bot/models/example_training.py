# models/example_training.py
"""
Esempio di addestramento del modello ML (eseguire dalla cartella del progetto):
    python -m models.example_training
Equivale a:  python main.py --mode train --years 8
"""
from data.loaders import load_prices
from models.train import ModelTrainer
from models.predict import ModelPredictor


def main():
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC']
    print("Scaricamento dati...")
    data = load_prices(symbols, years=8)
    if not data:
        raise SystemExit("Nessun dato scaricato: controlla la connessione internet")

    trainer = ModelTrainer(model_type='xgboost')
    meta = trainer.train(data)
    print(f"Training {meta['train_period']}  |  Test {meta['test_period']}")
    for k, v in meta['test_metrics'].items():
        print(f"  {k:24s} {v:+.4f}")

    path = './saved_models/xgboost'
    trainer.save(path)

    predictor = ModelPredictor(path)
    for symbol, df in data.items():
        r = predictor.predict_symbol(df)
        print(f"{symbol}: gradimento {r['gradimento']:+.3f}")


if __name__ == '__main__':
    main()
