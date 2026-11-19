# tests/conftest.py
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.synthetic import make_prices, make_universe  # noqa: E402


@pytest.fixture(scope='session')
def universe():
    return make_universe(n=700)


@pytest.fixture(scope='session')
def prices():
    return make_prices(n=400, seed=7)['close']
