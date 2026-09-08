# example_training.py
import pandas as pd
import yfinance as yf
from models.trainer import ModelTrainer
from models.data_pipeline import DataPipeline
from models.ensemble_model import EnsembleModel

def main():
    # 1. Scarica i dati
    print("Scaricamento dati...")
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC']
    
    data = []
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='5y')
        df['symbol'] = symbol
        data.append(df)
    
    df = pd.concat(data)
    print(f"Dati caricati: {len(df)} righe")
    
    # 2. Prepara i dati per il training
    print("\nPreparazione dati...")
    pipeline = DataPipeline()
    features_df = pipeline.create_features(df)
    X, y = pipeline.prepare_data(features_df)
    
    print(f"Feature: {X.shape[1]}, Samples: {X.shape[0]}")
    
    # 3. Addestra il modello
    print("\nAddestramento modello...")
    trainer = ModelTrainer(df, model_type='ensemble')
    train_metrics = trainer.train()
    
    # 4. Visualizza i risultati
    print("\nMetriche di addestramento:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nMetriche di valutazione:")
    if trainer.metrics:
        for key, value in trainer.metrics.__dict__.items():
            print(f"  {key}: {value:.4f}")
    
    # 5. Feature importance
    print("\nFeature Importance (Top 10):")
    importance = trainer.model.get_feature_importance()
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feature, imp in sorted_imp[:10]:
        print(f"  {feature}: {imp:.4f}")
    
    # 6. Salva il modello
    print("\nSalvataggio modello...")
    trainer.save('./saved_models/ensemble_model')
    print("Modello salvato!")
    
    # 7. Test su nuovi dati
    print("\nTest su nuovi dati...")
    # Scarica nuovi dati
    new_data = yf.download('AAPL', period='1mo')
    new_features = pipeline.create_features(new_data)
    X_new = pipeline.transform(new_features)
    
    predictions = trainer.predict(X_new[-5:])
    print(f"Ultime 5 predizioni: {predictions}")
    
    # 8. Esegui il modello in produzione
    print("\nPredizioni in tempo reale:")
    model = trainer.model
    
    for i, pred in enumerate(predictions):
        if pred > 0.5:
            signal = "STRONG BUY"
        elif pred > 0.2:
            signal = "BUY"
        elif pred < -0.5:
            signal = "STRONG SELL"
        elif pred < -0.2:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        print(f"  {i+1}: Gradimento = {pred:.3f} -> {signal}")

if __name__ == "__main__":
    main()