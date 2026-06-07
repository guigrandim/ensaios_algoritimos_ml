import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clf_data():
    np.random.seed(42)
    n = 300
    X = pd.DataFrame(np.random.randn(n, 8), columns=[f'f{i}' for i in range(8)])
    y = pd.DataFrame(((X['f0'] + X['f1']) > 0).astype(int), columns=['target'])
    tr, va = n // 2, n * 3 // 4
    return X.iloc[:tr], y.iloc[:tr], X.iloc[tr:va], y.iloc[tr:va], X.iloc[va:], y.iloc[va:]


@pytest.fixture
def reg_data():
    np.random.seed(42)
    n = 300
    X = pd.DataFrame(np.random.randn(n, 8), columns=[f'f{i}' for i in range(8)])
    y = pd.DataFrame(X['f0'] * 3 + X['f1'] * 1.5 + np.random.randn(n) * 0.5, columns=['target'])
    tr, va = n // 2, n * 3 // 4
    return X.iloc[:tr], y.iloc[:tr], X.iloc[tr:va], y.iloc[tr:va], X.iloc[va:], y.iloc[va:]


@pytest.fixture
def cluster_data():
    np.random.seed(42)
    n = 150
    return pd.DataFrame(np.random.randn(n, 4), columns=[f'f{i}' for i in range(4)])
