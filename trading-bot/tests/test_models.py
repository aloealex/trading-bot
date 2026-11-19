# tests/test_models.py
import numpy as np
import pandas as pd
import pytest

from models.features import build_features, build_dataset, FEATURE_NAMES
from models.train import ModelTrainer
from models.predict import ModelPredictor
from tests.synthetic import make_prices


def test_feature_causali():
    """Le feature alla data t non devono cambiare se cambia il futuro."""
    df = make_prices(n=400)
    t = df.index[300]
    a = build_features(df).loc[t]
    df2 = df.copy()
    df2.loc[df2.index > t, ['open', 'high', 'low', 'close']] *= 3
    b = build_features(df2).loc[t]
    pd.testing.assert_series_equal(a, b)


def test_dataset_allineato(universe):
    ds = build_dataset(universe)
    assert list(ds.columns) == FEATURE_NAMES + ['target', 'symbol']
    assert not ds[FEATURE_NAMES + ['target']].isnull().any().any()
    df = universe['AAA']
    row = ds[ds.symbol == 'AAA'].iloc[0]
    t = ds[ds.symbol == 'AAA'].index[0]
    i = df.index.get_loc(t)
    assert row['target'] == pytest.approx(df['close'].iloc[i + 5] / df['close'].iloc[i] - 1)


def test_train_salva_carica_predici(universe, tmp_path):
    trainer = ModelTrainer(model_type='xgboost', model_params={'n_estimators': 100})
    meta = trainer.train(universe)
    # il periodo di test inizia dopo la fine del training (+ buco di 5 giorni)
    assert pd.Timestamp(meta['test_period'][0]) > pd.Timestamp(meta['train_period'][1])
    trainer.save(str(tmp_path))

    p = ModelPredictor(str(tmp_path))
    r = p.predict_symbol(universe['BBB'])
    assert 'error' not in r and -1 <= r['gradimento'] <= 1
    series = p.predict_series(universe['BBB'])
    assert series.abs().max() <= 1 and series.std() > 0.1  # i valori superano davvero le soglie


def test_predict_storico_corto_non_crasha(universe, tmp_path):
    trainer = ModelTrainer(model_type='xgboost', model_params={'n_estimators': 50})
    trainer.train(universe)
    trainer.save(str(tmp_path))
    r = ModelPredictor(str(tmp_path)).predict_symbol(universe['AAA'].iloc[:50])
    assert r['gradimento'] == 0.0 and 'error' in r


def test_ensemble_e_rete_neurale(universe, tmp_path):
    torch = pytest.importorskip('torch')
    small = {s: df for s, df in list(universe.items())[:2]}
    trainer = ModelTrainer(model_type='ensemble')
    meta = trainer.train(small)
    assert set(meta['train_metrics']['weights']) == {'xgboost', 'neural'}
    trainer.save(str(tmp_path))
    r = ModelPredictor(str(tmp_path)).predict_symbol(small['AAA'])
    assert 'error' not in r
