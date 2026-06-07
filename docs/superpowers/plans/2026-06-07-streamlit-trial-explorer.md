# Streamlit Trial Explorer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit app with one page per ML category (Classification, Regression, Clustering) that re-executes models in real time, lets the user tune hyperparameters via sliders, and shows a comparative metrics table plus a performance-vs-hyperparameter curve.

**Architecture:** Pure ML logic lives in `experiments/` (no Streamlit dependency — fully testable). Streamlit pages in `pages/` call `experiments/` functions and render results with Plotly. Datasets are loaded via `@st.cache_data` in each page.

**Tech Stack:** Python 3.11, Streamlit, Plotly, scikit-learn, XGBoost, LightGBM, pandas, numpy, pytest

---

## File Map

| File | Role |
|------|------|
| `app.py` | Home page — project description and navigation hint |
| `pages/1_Classificacao.py` | Classification UI |
| `pages/2_Regressao.py` | Regression UI |
| `pages/3_Clusterizacao.py` | Clustering UI |
| `experiments/__init__.py` | Empty package marker |
| `experiments/classification.py` | ML logic: KNN, Decision Tree, Random Forest, Logistic Regression |
| `experiments/regression.py` | ML logic: 12 regression algorithms |
| `experiments/clustering.py` | ML logic: KMeans, Affinity Propagation |
| `tests/conftest.py` | Shared pytest fixtures (synthetic datasets) |
| `tests/test_classification.py` | Unit tests for classification module |
| `tests/test_regression.py` | Unit tests for regression module |
| `tests/test_clustering.py` | Unit tests for clustering module |

---

## Task 1: Setup — directories, dependencies, package skeleton

**Files:**
- Create: `experiments/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: (no existing files)

- [ ] **Step 1.1: Install dependencies**

```bash
pip install streamlit plotly pytest
```

Expected: packages install without errors. Verify with `streamlit --version` and `pytest --version`.

- [ ] **Step 1.2: Create directory structure**

```bash
mkdir experiments pages tests
```

- [ ] **Step 1.3: Create `experiments/__init__.py`**

```python
```
(empty file)

- [ ] **Step 1.4: Create `tests/__init__.py`**

```python
```
(empty file)

- [ ] **Step 1.5: Create `tests/conftest.py` with shared fixtures**

```python
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
```

- [ ] **Step 1.6: Verify fixtures load**

```bash
cd D:/repos/projetos/ml_trials_algorithm
python -c "import tests.conftest; print('ok')"
```

Expected: prints `ok` with no errors.

---

## Task 2: `experiments/classification.py`

**Files:**
- Create: `experiments/classification.py`
- Create: `tests/test_classification.py`

- [ ] **Step 2.1: Write failing tests**

Create `tests/test_classification.py`:

```python
import pytest
from experiments.classification import run_trial


_EXPECTED_CONJUNTOS = [
    'Treino Default', 'Validação Default',
    'Treino Tunado', 'Validação Tunada', 'Teste Final'
]


def test_knn_structure(clf_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = clf_data
    result = run_trial('knn', {'k': 5}, X_tr, y_tr, X_v, y_v, X_te, y_te)
    assert set(result.keys()) == {'quadro', 'curva', 'param_name'}
    assert result['quadro'].shape == (5, 5)
    assert list(result['quadro']['Conjunto']) == _EXPECTED_CONJUNTOS
    assert result['param_name'] == 'k'
    assert len(result['curva']) == 20  # k = 1..20
    assert set(result['curva'].columns) >= {'Valor', 'Acc_train', 'Acc_val', 'F1_train', 'F1_val'}


def test_decision_tree_structure(clf_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = clf_data
    result = run_trial('decision_tree', {'max_depth': 5}, X_tr, y_tr, X_v, y_v, X_te, y_te)
    assert result['quadro'].shape == (5, 5)
    assert result['param_name'] == 'max_depth'
    assert len(result['curva']) == 20  # max_depth = 1..20


def test_random_forest_structure(clf_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = clf_data
    result = run_trial('random_forest', {'max_depth': 5, 'n_estimators': 50},
                       X_tr, y_tr, X_v, y_v, X_te, y_te)
    assert result['quadro'].shape == (5, 5)
    assert result['param_name'] == 'max_depth'
    assert len(result['curva']) == 20


def test_logistic_regression_structure(clf_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = clf_data
    result = run_trial('logistic_regression', {'C': 1.0},
                       X_tr, y_tr, X_v, y_v, X_te, y_te)
    assert result['quadro'].shape == (5, 5)
    assert result['param_name'] == 'C'
    assert len(result['curva']) == 6  # C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def test_metrics_in_valid_range(clf_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = clf_data
    result = run_trial('knn', {'k': 3}, X_tr, y_tr, X_v, y_v, X_te, y_te)
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        assert result['quadro'][col].between(0.0, 1.0).all(), f"{col} out of [0,1]"


def test_unknown_algorithm_raises(clf_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = clf_data
    with pytest.raises(ValueError, match="Unknown algorithm"):
        run_trial('svm', {}, X_tr, y_tr, X_v, y_v, X_te, y_te)
```

- [ ] **Step 2.2: Run tests — expect FAIL**

```bash
cd D:/repos/projetos/ml_trials_algorithm
pytest tests/test_classification.py -v
```

Expected: `ModuleNotFoundError: No module named 'experiments.classification'`

- [ ] **Step 2.3: Implement `experiments/classification.py`**

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as mt


_SWEEP_RANGES = {
    'knn': list(range(1, 21)),
    'decision_tree': list(range(1, 21)),
    'random_forest': list(range(1, 21)),
    'logistic_regression': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
}

_PARAM_NAMES = {
    'knn': 'k',
    'decision_tree': 'max_depth',
    'random_forest': 'max_depth',
    'logistic_regression': 'C',
}

_DEFAULTS = {
    'knn': {'k': 5},
    'decision_tree': {'max_depth': None},
    'random_forest': {'max_depth': None, 'n_estimators': 100},
    'logistic_regression': {'C': 1.0},
}


def _build_model(algorithm: str, params: dict):
    if algorithm == 'knn':
        return KNeighborsClassifier(n_neighbors=params.get('k', 5))
    if algorithm == 'decision_tree':
        return DecisionTreeClassifier(max_depth=params.get('max_depth'), random_state=42)
    if algorithm == 'random_forest':
        return RandomForestClassifier(
            max_depth=params.get('max_depth'),
            n_estimators=params.get('n_estimators', 100),
            random_state=42, n_jobs=-1,
        )
    if algorithm == 'logistic_regression':
        return LogisticRegression(C=params.get('C', 1.0), max_iter=1000, random_state=42)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _clf_metrics(y_true, y_pred) -> dict:
    return {
        'Accuracy':  round(float(mt.accuracy_score(y_true, y_pred)), 4),
        'Precision': round(float(mt.precision_score(y_true, y_pred, zero_division=0)), 4),
        'Recall':    round(float(mt.recall_score(y_true, y_pred, zero_division=0)), 4),
        'F1-Score':  round(float(mt.f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def run_trial(algorithm: str, params: dict,
              X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    if algorithm not in _PARAM_NAMES:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    def ravel(y):
        return y.values.ravel() if hasattr(y, 'values') else np.array(y).ravel()

    y_tr, y_v, y_te = ravel(y_train), ravel(y_val), ravel(y_test)

    # Default model — train+val metrics
    m_def = _build_model(algorithm, _DEFAULTS[algorithm])
    m_def.fit(X_train, y_tr)
    m_train_def = _clf_metrics(y_tr, m_def.predict(X_train))
    m_val_def   = _clf_metrics(y_v,  m_def.predict(X_val))

    # Tuned model on train only — to fill quadro tuned rows
    m_t = _build_model(algorithm, params)
    m_t.fit(X_train, y_tr)
    m_train_t = _clf_metrics(y_tr, m_t.predict(X_train))
    m_val_t   = _clf_metrics(y_v,  m_t.predict(X_val))

    # Final model: train+val → test
    X_final = (pd.concat([X_train, X_val])
               if isinstance(X_train, pd.DataFrame)
               else np.vstack([X_train, X_val]))
    y_final = np.concatenate([y_tr, y_v])
    m_final = _build_model(algorithm, params)
    m_final.fit(X_final, y_final)
    m_test = _clf_metrics(y_te, m_final.predict(X_test))

    quadro = pd.DataFrame([
        {'Conjunto': 'Treino Default',    **m_train_def},
        {'Conjunto': 'Validação Default', **m_val_def},
        {'Conjunto': 'Treino Tunado',     **m_train_t},
        {'Conjunto': 'Validação Tunada',  **m_val_t},
        {'Conjunto': 'Teste Final',       **m_test},
    ])

    # Performance curve — sweep main hyperparameter
    param_name   = _PARAM_NAMES[algorithm]
    sweep_values = _SWEEP_RANGES[algorithm]
    rows = []
    for val in sweep_values:
        sp = dict(params)
        sp[param_name] = val
        m = _build_model(algorithm, sp)
        m.fit(X_train, y_tr)
        tr = _clf_metrics(y_tr, m.predict(X_train))
        va = _clf_metrics(y_v,  m.predict(X_val))
        rows.append({
            'Valor':      val,
            'Acc_train':  tr['Accuracy'],  'Acc_val':  va['Accuracy'],
            'Prec_train': tr['Precision'], 'Prec_val': va['Precision'],
            'Rec_train':  tr['Recall'],    'Rec_val':  va['Recall'],
            'F1_train':   tr['F1-Score'],  'F1_val':   va['F1-Score'],
        })
    curva = pd.DataFrame(rows)

    return {'quadro': quadro, 'curva': curva, 'param_name': param_name}
```

- [ ] **Step 2.4: Run tests — expect PASS**

```bash
pytest tests/test_classification.py -v
```

Expected: all 6 tests PASS.

---

## Task 3: `experiments/regression.py`

**Files:**
- Create: `experiments/regression.py`
- Create: `tests/test_regression.py`

- [ ] **Step 3.1: Write failing tests**

Create `tests/test_regression.py`:

```python
import pytest
from experiments.regression import run_trial


_EXPECTED_CONJUNTOS = [
    'Treino Default', 'Validação Default',
    'Treino Tunado', 'Validação Tunada', 'Teste Final'
]


def test_linear_regression_no_curve(reg_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = reg_data
    result = run_trial('linear_regression', {}, X_tr, y_tr, X_v, y_v, X_te, y_te)
    assert result['quadro'].shape == (5, 5)
    assert list(result['quadro']['Conjunto']) == _EXPECTED_CONJUNTOS
    assert result['curva'] is None
    assert result['param_name'] is None


def test_lasso_structure(reg_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = reg_data
    result = run_trial('lasso', {'alpha': 0.01}, X_tr, y_tr, X_v, y_v, X_te, y_te)
    assert result['quadro'].shape == (5, 5)
    assert result['param_name'] == 'alpha'
    assert len(result['curva']) == 7  # alpha in [0.0001..100.0]
    assert set(result['curva'].columns) >= {'Valor', 'R2_train', 'R2_val', 'RMSE_train', 'RMSE_val'}


def test_polynomial_structure(reg_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = reg_data
    result = run_trial('polynomial', {'degree': 2}, X_tr, y_tr, X_v, y_v, X_te, y_te)
    assert result['quadro'].shape == (5, 5)
    assert result['param_name'] == 'degree'
    assert len(result['curva']) == 5  # degree = 1..5


def test_decision_tree_structure(reg_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = reg_data
    result = run_trial('decision_tree', {'max_depth': 5}, X_tr, y_tr, X_v, y_v, X_te, y_te)
    assert result['quadro'].shape == (5, 5)
    assert len(result['curva']) == 20  # max_depth = 1..20


def test_xgboost_structure(reg_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = reg_data
    result = run_trial('xgboost', {'max_depth': 5, 'n_estimators': 100, 'learning_rate': 0.1},
                       X_tr, y_tr, X_v, y_v, X_te, y_te)
    assert result['quadro'].shape == (5, 5)
    assert result['param_name'] == 'n_estimators'
    assert len(result['curva']) == 6  # n_estimators 100..600 step 100


def test_r2_column_exists(reg_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = reg_data
    result = run_trial('ridge', {'alpha': 1.0}, X_tr, y_tr, X_v, y_v, X_te, y_te)
    assert set(result['quadro'].columns) == {'Conjunto', 'R²', 'RMSE', 'MAE', 'MAPE'}


def test_unknown_algorithm_raises(reg_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = reg_data
    with pytest.raises(ValueError, match="Unknown algorithm"):
        run_trial('svm', {}, X_tr, y_tr, X_v, y_v, X_te, y_te)
```

- [ ] **Step 3.2: Run tests — expect FAIL**

```bash
pytest tests/test_regression.py -v
```

Expected: `ModuleNotFoundError: No module named 'experiments.regression'`

- [ ] **Step 3.3: Implement `experiments/regression.py`**

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import metrics as mt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


_ALPHA_RANGE  = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
_DEGREE_RANGE = [1, 2, 3, 4, 5]
_DEPTH_RANGE  = list(range(1, 21))
_N_EST_RANGE  = list(range(100, 700, 100))

_PARAM_NAMES = {
    'linear_regression':      None,
    'lasso':                  'alpha',
    'ridge':                  'alpha',
    'elasticnet':             'alpha',
    'polynomial':             'degree',
    'polynomial_lasso':       'degree',
    'polynomial_ridge':       'degree',
    'polynomial_elasticnet':  'degree',
    'decision_tree':          'max_depth',
    'random_forest':          'max_depth',
    'xgboost':                'n_estimators',
    'lightgbm':               'n_estimators',
}

_SWEEP_RANGES = {
    'lasso':                 _ALPHA_RANGE,
    'ridge':                 _ALPHA_RANGE,
    'elasticnet':            _ALPHA_RANGE,
    'polynomial':            _DEGREE_RANGE,
    'polynomial_lasso':      _DEGREE_RANGE,
    'polynomial_ridge':      _DEGREE_RANGE,
    'polynomial_elasticnet': _DEGREE_RANGE,
    'decision_tree':         _DEPTH_RANGE,
    'random_forest':         _DEPTH_RANGE,
    'xgboost':               _N_EST_RANGE,
    'lightgbm':              _N_EST_RANGE,
}

_DEFAULTS = {
    'linear_regression':     {},
    'lasso':                 {'alpha': 1.0},
    'ridge':                 {'alpha': 1.0},
    'elasticnet':            {'alpha': 1.0, 'l1_ratio': 0.5},
    'polynomial':            {'degree': 1},
    'polynomial_lasso':      {'degree': 1, 'alpha': 1.0},
    'polynomial_ridge':      {'degree': 1, 'alpha': 1.0},
    'polynomial_elasticnet': {'degree': 1, 'alpha': 1.0, 'l1_ratio': 0.5},
    'decision_tree':         {'max_depth': None},
    'random_forest':         {'max_depth': None, 'n_estimators': 100},
    'xgboost':               {'max_depth': 6, 'n_estimators': 100, 'learning_rate': 0.3},
    'lightgbm':              {'max_depth': -1, 'n_estimators': 100, 'learning_rate': 0.1,
                              'num_leaves': 31, 'min_child_samples': 20},
}


def _build_model(algorithm: str, params: dict):
    if algorithm == 'linear_regression':
        return LinearRegression()
    if algorithm == 'lasso':
        return Lasso(alpha=params.get('alpha', 1.0), max_iter=5000)
    if algorithm == 'ridge':
        return Ridge(alpha=params.get('alpha', 1.0))
    if algorithm == 'elasticnet':
        return ElasticNet(alpha=params.get('alpha', 1.0),
                          l1_ratio=params.get('l1_ratio', 0.5), max_iter=5000)
    if algorithm == 'polynomial':
        return Pipeline([
            ('poly', PolynomialFeatures(degree=params.get('degree', 1))),
            ('model', LinearRegression()),
        ])
    if algorithm == 'polynomial_lasso':
        return Pipeline([
            ('poly', PolynomialFeatures(degree=params.get('degree', 1))),
            ('model', Lasso(alpha=params.get('alpha', 1.0), max_iter=5000)),
        ])
    if algorithm == 'polynomial_ridge':
        return Pipeline([
            ('poly', PolynomialFeatures(degree=params.get('degree', 1))),
            ('model', Ridge(alpha=params.get('alpha', 1.0))),
        ])
    if algorithm == 'polynomial_elasticnet':
        return Pipeline([
            ('poly', PolynomialFeatures(degree=params.get('degree', 1))),
            ('model', ElasticNet(alpha=params.get('alpha', 1.0),
                                 l1_ratio=params.get('l1_ratio', 0.5), max_iter=5000)),
        ])
    if algorithm == 'decision_tree':
        return DecisionTreeRegressor(max_depth=params.get('max_depth'), random_state=42)
    if algorithm == 'random_forest':
        return RandomForestRegressor(
            max_depth=params.get('max_depth'),
            n_estimators=params.get('n_estimators', 100),
            random_state=42, n_jobs=-1,
        )
    if algorithm == 'xgboost':
        return XGBRegressor(
            max_depth=params.get('max_depth', 6),
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=42, verbosity=0,
        )
    if algorithm == 'lightgbm':
        return LGBMRegressor(
            max_depth=params.get('max_depth', -1),
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            num_leaves=params.get('num_leaves', 31),
            min_child_samples=params.get('min_child_samples', 20),
            random_state=42, verbosity=-1,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _reg_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mt.mean_squared_error(y_true, y_pred)))
    return {
        'R²':   round(float(mt.r2_score(y_true, y_pred)), 4),
        'RMSE': round(rmse, 4),
        'MAE':  round(float(mt.mean_absolute_error(y_true, y_pred)), 4),
        'MAPE': round(float(mt.mean_absolute_percentage_error(y_true, y_pred)), 4),
    }


def run_trial(algorithm: str, params: dict,
              X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    if algorithm not in _PARAM_NAMES:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    def ravel(y):
        return y.values.ravel() if hasattr(y, 'values') else np.array(y).ravel()

    y_tr, y_v, y_te = ravel(y_train), ravel(y_val), ravel(y_test)

    m_def = _build_model(algorithm, _DEFAULTS[algorithm])
    m_def.fit(X_train, y_tr)
    m_train_def = _reg_metrics(y_tr, m_def.predict(X_train))
    m_val_def   = _reg_metrics(y_v,  m_def.predict(X_val))

    m_t = _build_model(algorithm, params)
    m_t.fit(X_train, y_tr)
    m_train_t = _reg_metrics(y_tr, m_t.predict(X_train))
    m_val_t   = _reg_metrics(y_v,  m_t.predict(X_val))

    X_final = (pd.concat([X_train, X_val])
               if isinstance(X_train, pd.DataFrame)
               else np.vstack([X_train, X_val]))
    y_final = np.concatenate([y_tr, y_v])
    m_final = _build_model(algorithm, params)
    m_final.fit(X_final, y_final)
    m_test = _reg_metrics(y_te, m_final.predict(X_test))

    quadro = pd.DataFrame([
        {'Conjunto': 'Treino Default',    **m_train_def},
        {'Conjunto': 'Validação Default', **m_val_def},
        {'Conjunto': 'Treino Tunado',     **m_train_t},
        {'Conjunto': 'Validação Tunada',  **m_val_t},
        {'Conjunto': 'Teste Final',       **m_test},
    ])

    param_name = _PARAM_NAMES[algorithm]
    if param_name is None:
        return {'quadro': quadro, 'curva': None, 'param_name': None}

    sweep_values = _SWEEP_RANGES[algorithm]
    rows = []
    for val in sweep_values:
        sp = dict(params)
        sp[param_name] = val
        m = _build_model(algorithm, sp)
        m.fit(X_train, y_tr)
        tr = _reg_metrics(y_tr, m.predict(X_train))
        va = _reg_metrics(y_v,  m.predict(X_val))
        rows.append({
            'Valor':      val,
            'R2_train':   tr['R²'],   'R2_val':   va['R²'],
            'RMSE_train': tr['RMSE'], 'RMSE_val': va['RMSE'],
            'MAE_train':  tr['MAE'],  'MAE_val':  va['MAE'],
            'MAPE_train': tr['MAPE'], 'MAPE_val': va['MAPE'],
        })
    curva = pd.DataFrame(rows)

    return {'quadro': quadro, 'curva': curva, 'param_name': param_name}
```

- [ ] **Step 3.4: Run tests — expect PASS**

```bash
pytest tests/test_regression.py -v
```

Expected: all 7 tests PASS.

---

## Task 4: `experiments/clustering.py`

**Files:**
- Create: `experiments/clustering.py`
- Create: `tests/test_clustering.py`

- [ ] **Step 4.1: Write failing tests**

Create `tests/test_clustering.py`:

```python
import pytest
from experiments.clustering import run_trial


def test_kmeans_structure(cluster_data):
    result = run_trial('kmeans', {'k': 3}, cluster_data)
    assert set(result.keys()) == {'quadro', 'curva', 'param_name'}
    assert result['quadro'].shape == (2, 3)
    assert list(result['quadro']['Configuração']) == ['Default', 'Tunado']
    assert set(result['quadro'].columns) == {'Configuração', 'N° Clusters', 'Silhouette Score'}
    assert result['param_name'] == 'k'
    assert len(result['curva']) <= 9  # sweep k=2..10, some may be dropped if N/A


def test_affinity_propagation_structure(cluster_data):
    result = run_trial('affinity_propagation', {'preference': -100}, cluster_data)
    assert result['quadro'].shape == (2, 3)
    assert result['param_name'] == 'preference'
    assert len(result['curva']) == 7  # 7 preference values


def test_single_cluster_silhouette_is_na(cluster_data):
    # preference=-1000 forces a single cluster → Silhouette must be 'N/A'
    result = run_trial('affinity_propagation', {'preference': -1000}, cluster_data)
    tuned_sil = result['quadro'].loc[
        result['quadro']['Configuração'] == 'Tunado', 'Silhouette Score'
    ].values[0]
    assert tuned_sil == 'N/A'


def test_kmeans_curva_has_silhouette_column(cluster_data):
    result = run_trial('kmeans', {'k': 3}, cluster_data)
    assert 'Silhouette' in result['curva'].columns
    assert 'Valor' in result['curva'].columns


def test_unknown_algorithm_raises(cluster_data):
    with pytest.raises(ValueError, match="Unknown algorithm"):
        run_trial('dbscan', {}, cluster_data)
```

- [ ] **Step 4.2: Run tests — expect FAIL**

```bash
pytest tests/test_clustering.py -v
```

Expected: `ModuleNotFoundError: No module named 'experiments.clustering'`

- [ ] **Step 4.3: Implement `experiments/clustering.py`**

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn import metrics as mt


_KMEANS_SWEEP  = list(range(2, 11))                            # k = 2..10
_AP_SWEEP      = [-50, -100, -200, -300, -500, -700, -1000]


def _silhouette(X_arr, labels):
    unique = set(labels) - {-1}
    if len(unique) < 2:
        return 'N/A'
    return round(float(mt.silhouette_score(X_arr, labels)), 4)


def run_trial(algorithm: str, params: dict, X) -> dict:
    if algorithm not in ('kmeans', 'affinity_propagation'):
        raise ValueError(f"Unknown algorithm: {algorithm}")

    X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)

    if algorithm == 'kmeans':
        # Default
        m_def = KMeans(random_state=42)
        labels_def = m_def.fit_predict(X_arr)
        n_def  = len(set(labels_def))
        sil_def = _silhouette(X_arr, labels_def)

        # Tuned
        k = params.get('k', 3)
        m_tuned = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels_tuned = m_tuned.fit_predict(X_arr)
        sil_tuned = _silhouette(X_arr, labels_tuned)

        quadro = pd.DataFrame([
            {'Configuração': 'Default', 'N° Clusters': n_def, 'Silhouette Score': sil_def},
            {'Configuração': 'Tunado',  'N° Clusters': k,     'Silhouette Score': sil_tuned},
        ])

        rows = []
        for k_val in _KMEANS_SWEEP:
            m = KMeans(n_clusters=k_val, random_state=42)
            sil = _silhouette(X_arr, m.fit_predict(X_arr))
            if sil != 'N/A':
                rows.append({'Valor': k_val, 'Silhouette': float(sil)})
        curva = pd.DataFrame(rows)

        return {'quadro': quadro, 'curva': curva, 'param_name': 'k'}

    # affinity_propagation
    m_def = AffinityPropagation(random_state=42)
    labels_def = m_def.fit_predict(X_arr)
    n_def   = len(m_def.cluster_centers_indices_)
    sil_def = _silhouette(X_arr, labels_def)

    pref = params.get('preference', -100)
    m_tuned = AffinityPropagation(preference=pref, damping=0.6, max_iter=1000, random_state=42)
    labels_tuned = m_tuned.fit_predict(X_arr)
    n_tuned   = len(m_tuned.cluster_centers_indices_)
    sil_tuned = _silhouette(X_arr, labels_tuned)

    quadro = pd.DataFrame([
        {'Configuração': 'Default', 'N° Clusters': n_def,   'Silhouette Score': sil_def},
        {'Configuração': 'Tunado',  'N° Clusters': n_tuned, 'Silhouette Score': sil_tuned},
    ])

    rows = []
    for pref_val in _AP_SWEEP:
        m = AffinityPropagation(preference=pref_val, damping=0.6, max_iter=1000, random_state=42)
        sil = _silhouette(X_arr, m.fit_predict(X_arr))
        rows.append({'Valor': pref_val, 'Silhouette': sil})
    curva = pd.DataFrame(rows)

    return {'quadro': quadro, 'curva': curva, 'param_name': 'preference'}
```

- [ ] **Step 4.4: Run all tests — expect all PASS**

```bash
pytest tests/ -v
```

Expected: all tests in all 3 files PASS.

---

## Task 5: `app.py` — Home page

**Files:**
- Create: `app.py`

- [ ] **Step 5.1: Create `app.py`**

```python
import streamlit as st

st.set_page_config(page_title="ML Trial Explorer", page_icon="🔬", layout="wide")

st.title("ML Trial Explorer")
st.markdown("""
Ensaios experimentais comparativos de algoritmos de machine learning organizados em três categorias.

**Navegue pelas páginas no menu lateral:**

| Página | Algoritmos |
|--------|-----------|
| Classificação | KNN, Decision Tree, Random Forest, Logistic Regression |
| Regressão | Linear, Lasso, Ridge, ElasticNet, Polynomial (×4), Decision Tree, Random Forest, XGBoost, LightGBM |
| Clusterização | KMeans, Affinity Propagation |

**Como usar:**
1. Selecione a categoria na barra lateral
2. Escolha o algoritmo e ajuste os hiperparâmetros
3. Clique em **Executar Ensaio**
4. Analise o quadro comparativo e a curva de performance

---
*Projeto: Data Money — Ensaios de ML*
""")
```

- [ ] **Step 5.2: Verify app launches**

```bash
cd D:/repos/projetos/ml_trials_algorithm
streamlit run app.py
```

Expected: browser opens showing the home page with the description table.

---

## Task 6: `pages/1_Classificacao.py`

**Files:**
- Create: `pages/1_Classificacao.py`

- [ ] **Step 6.1: Create classification page**

```python
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from experiments.classification import run_trial

st.set_page_config(page_title="Classificação", layout="wide")
st.title("Ensaios — Classificação")

_ALGO_LABELS = {
    'knn':                  'KNN',
    'decision_tree':        'Decision Tree',
    'random_forest':        'Random Forest',
    'logistic_regression':  'Logistic Regression',
}

_METRIC_COLS = {
    'Accuracy':  ('Acc_train',  'Acc_val'),
    'Precision': ('Prec_train', 'Prec_val'),
    'Recall':    ('Rec_train',  'Rec_val'),
    'F1-Score':  ('F1_train',   'F1_val'),
}


@st.cache_data
def load_data():
    root = Path(__file__).resolve().parent.parent
    base = root / 'dataset' / 'classification_datasets'
    return (
        pd.read_csv(base / 'a_traninig'  / 'X_training.csv'),
        pd.read_csv(base / 'a_traninig'  / 'y_training.csv'),
        pd.read_csv(base / 'b_validation' / 'X_validation.csv'),
        pd.read_csv(base / 'b_validation' / 'y_validation.csv'),
        pd.read_csv(base / 'c_test'       / 'X_test.csv'),
        pd.read_csv(base / 'c_test'       / 'y_test.csv'),
    )


try:
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
except FileNotFoundError as e:
    st.error(f"Dataset não encontrado: {e}")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuração do Ensaio")
    algorithm = st.selectbox(
        "Algoritmo",
        list(_ALGO_LABELS.keys()),
        format_func=_ALGO_LABELS.get,
    )
    params: dict = {}
    if algorithm == 'knn':
        params['k'] = st.slider('k (vizinhos)', 1, 20, 5)
    elif algorithm == 'decision_tree':
        params['max_depth'] = st.slider('max_depth', 1, 20, 10)
    elif algorithm == 'random_forest':
        params['max_depth']     = st.slider('max_depth', 1, 20, 10)
        params['n_estimators']  = st.select_slider(
            'n_estimators', options=[50, 60, 70, 80, 90, 100], value=100)
    elif algorithm == 'logistic_regression':
        params['C'] = st.select_slider(
            'C (regularização)', options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
    run = st.button("Executar Ensaio", type="primary", use_container_width=True)

# ── Main ─────────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Executando ensaio..."):
        result = run_trial(algorithm, params, X_train, y_train, X_val, y_val, X_test, y_test)

    quadro     = result['quadro']
    curva      = result['curva']
    param_name = result['param_name']

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Quadro Comparativo")
        fmt = {c: '{:.4f}' for c in ['Accuracy', 'Precision', 'Recall', 'F1-Score']}
        st.dataframe(quadro.style.format(fmt), use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Curva de Performance")
        metric = st.selectbox("Métrica", list(_METRIC_COLS.keys()))
        train_col, val_col = _METRIC_COLS[metric]

        param_key = 'k' if algorithm == 'knn' else param_name
        highlight = params.get(param_key)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curva['Valor'], y=curva[train_col],
            mode='lines+markers', name='Treino',
        ))
        fig.add_trace(go.Scatter(
            x=curva['Valor'], y=curva[val_col],
            mode='lines+markers', name='Validação',
        ))
        if highlight in curva['Valor'].values:
            row = curva[curva['Valor'] == highlight].iloc[0]
            fig.add_trace(go.Scatter(
                x=[highlight, highlight],
                y=[row[train_col], row[val_col]],
                mode='markers',
                marker=dict(size=14, color='red', symbol='star'),
                name='Selecionado',
            ))
        fig.update_layout(
            xaxis_title=param_name,
            yaxis_title=metric,
            legend=dict(orientation='h', yanchor='bottom', y=1.02),
            height=380,
        )
        st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 6.2: Smoke-test the classification page**

With the app running (`streamlit run app.py`):
1. Navigate to "Classificação"
2. Select "Random Forest", set `max_depth=10`, `n_estimators=100`
3. Click "Executar Ensaio"

Expected: quadro comparativo with 5 rows appears, curve shows 20 points (max_depth 1–20), red star at depth=10.

---

## Task 7: `pages/2_Regressao.py`

**Files:**
- Create: `pages/2_Regressao.py`

- [ ] **Step 7.1: Create regression page**

```python
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from experiments.regression import run_trial

st.set_page_config(page_title="Regressão", layout="wide")
st.title("Ensaios — Regressão")

_ALGO_LABELS = {
    'linear_regression':     'Linear Regression',
    'lasso':                 'Lasso',
    'ridge':                 'Ridge',
    'elasticnet':            'ElasticNet',
    'polynomial':            'Polynomial',
    'polynomial_lasso':      'Polynomial + Lasso',
    'polynomial_ridge':      'Polynomial + Ridge',
    'polynomial_elasticnet': 'Polynomial + ElasticNet',
    'decision_tree':         'Decision Tree',
    'random_forest':         'Random Forest',
    'xgboost':               'XGBoost',
    'lightgbm':              'LightGBM',
}

_METRIC_COLS = {
    'R²':   ('R2_train',   'R2_val'),
    'RMSE': ('RMSE_train', 'RMSE_val'),
    'MAE':  ('MAE_train',  'MAE_val'),
    'MAPE': ('MAPE_train', 'MAPE_val'),
}


@st.cache_data
def load_data():
    root = Path(__file__).resolve().parent.parent
    base = root / 'dataset' / 'regression_datasets'
    return (
        pd.read_csv(base / 'a_traninig'  / 'X_training.csv'),
        pd.read_csv(base / 'a_traninig'  / 'y_training.csv'),
        pd.read_csv(base / 'b_validation' / 'X_validation.csv'),
        pd.read_csv(base / 'b_validation' / 'y_validation.csv'),
        pd.read_csv(base / 'c_test'       / 'X_test.csv'),
        pd.read_csv(base / 'c_test'       / 'y_test.csv'),
    )


try:
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
except FileNotFoundError as e:
    st.error(f"Dataset não encontrado: {e}")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuração do Ensaio")
    algorithm = st.selectbox(
        "Algoritmo",
        list(_ALGO_LABELS.keys()),
        format_func=_ALGO_LABELS.get,
    )
    params: dict = {}

    if algorithm == 'linear_regression':
        st.info("Regressão Linear não possui hiperparâmetros para ajuste.")

    elif algorithm in ('lasso', 'ridge'):
        params['alpha'] = st.select_slider(
            'alpha', options=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)

    elif algorithm == 'elasticnet':
        params['alpha']    = st.select_slider(
            'alpha', options=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
        params['l1_ratio'] = st.slider('l1_ratio', 0.0, 1.0, 0.5, step=0.1)

    elif algorithm in ('polynomial', 'polynomial_lasso', 'polynomial_ridge', 'polynomial_elasticnet'):
        params['degree'] = st.slider('degree', 1, 5, 2)
        if algorithm in ('polynomial_lasso', 'polynomial_ridge'):
            params['alpha'] = st.select_slider(
                'alpha', options=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
        elif algorithm == 'polynomial_elasticnet':
            params['alpha']    = st.select_slider(
                'alpha', options=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0], value=1.0)
            params['l1_ratio'] = st.slider('l1_ratio', 0.0, 1.0, 0.5, step=0.1)

    elif algorithm == 'decision_tree':
        params['max_depth'] = st.slider('max_depth', 1, 20, 5)

    elif algorithm == 'random_forest':
        params['max_depth']    = st.slider('max_depth', 1, 20, 10)
        params['n_estimators'] = st.select_slider(
            'n_estimators', options=[50, 100, 200, 300], value=100)

    elif algorithm in ('xgboost', 'lightgbm'):
        params['max_depth']     = st.slider('max_depth', 3, 10, 6)
        params['n_estimators']  = st.select_slider(
            'n_estimators', options=[100, 200, 300, 400, 500, 600], value=100)
        params['learning_rate'] = st.select_slider(
            'learning_rate', options=[0.01, 0.05, 0.1, 0.3], value=0.1)
        if algorithm == 'lightgbm':
            params['num_leaves']       = st.slider('num_leaves', 10, 100, 31)
            params['min_child_samples'] = st.slider('min_child_samples', 5, 50, 20)

    run = st.button("Executar Ensaio", type="primary", use_container_width=True)

# ── Main ─────────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Executando ensaio..."):
        result = run_trial(algorithm, params, X_train, y_train, X_val, y_val, X_test, y_test)

    quadro     = result['quadro']
    curva      = result['curva']
    param_name = result['param_name']

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Quadro Comparativo")
        fmt = {'R²': '{:.4f}', 'RMSE': '{:.4f}', 'MAE': '{:.4f}', 'MAPE': '{:.2%}'}
        st.dataframe(quadro.style.format(fmt), use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("Curva de Performance")
        if curva is None:
            st.info("Regressão Linear não possui hiperparâmetros — curva não disponível.")
        else:
            metric = st.selectbox("Métrica", list(_METRIC_COLS.keys()))
            train_col, val_col = _METRIC_COLS[metric]
            highlight = params.get(param_name)

            y_train_vals = curva[train_col].copy()
            y_val_vals   = curva[val_col].copy()
            y_label      = metric

            if metric == 'MAPE':
                y_train_vals = y_train_vals * 100
                y_val_vals   = y_val_vals   * 100
                y_label      = 'MAPE (%)'

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=curva['Valor'], y=y_train_vals,
                mode='lines+markers', name='Treino',
            ))
            fig.add_trace(go.Scatter(
                x=curva['Valor'], y=y_val_vals,
                mode='lines+markers', name='Validação',
            ))
            if highlight is not None and highlight in curva['Valor'].values:
                idx = curva[curva['Valor'] == highlight].index[0]
                fig.add_trace(go.Scatter(
                    x=[highlight, highlight],
                    y=[y_train_vals.iloc[idx], y_val_vals.iloc[idx]],
                    mode='markers',
                    marker=dict(size=14, color='red', symbol='star'),
                    name='Selecionado',
                ))
            fig.update_layout(
                xaxis_title=param_name,
                yaxis_title=y_label,
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 7.2: Smoke-test the regression page**

With the app running:
1. Navigate to "Regressão"
2. Select "XGBoost", `max_depth=7`, `n_estimators=100`, `learning_rate=0.1`
3. Click "Executar Ensaio"

Expected: quadro with 5 rows and 4 metrics (R², RMSE, MAE, MAPE in % format), curve shows 6 points (n_estimators 100–600).

---

## Task 8: `pages/3_Clusterizacao.py`

**Files:**
- Create: `pages/3_Clusterizacao.py`

- [ ] **Step 8.1: Create clustering page**

```python
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from experiments.clustering import run_trial

st.set_page_config(page_title="Clusterização", layout="wide")
st.title("Ensaios — Clusterização")

_ALGO_LABELS = {
    'kmeans':                'KMeans',
    'affinity_propagation':  'Affinity Propagation',
}


@st.cache_data
def load_data():
    root = Path(__file__).resolve().parent.parent
    base = root / 'dataset' / 'clusters_datasets' / 'a_traning'
    return pd.read_csv(base / 'X_dataset.csv')


try:
    X = load_data()
except FileNotFoundError as e:
    st.error(f"Dataset não encontrado: {e}")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuração do Ensaio")
    algorithm = st.selectbox(
        "Algoritmo",
        list(_ALGO_LABELS.keys()),
        format_func=_ALGO_LABELS.get,
    )
    params: dict = {}
    if algorithm == 'kmeans':
        params['k'] = st.slider('k (clusters)', 2, 10, 3)
    elif algorithm == 'affinity_propagation':
        params['preference'] = st.select_slider(
            'preference', options=[-50, -100, -200, -300, -500, -700, -1000], value=-100)
    run = st.button("Executar Ensaio", type="primary", use_container_width=True)

# ── Main ─────────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Executando ensaio..."):
        result = run_trial(algorithm, params, X)

    quadro     = result['quadro']
    curva      = result['curva']
    param_name = result['param_name']

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Quadro Comparativo")
        # Format Silhouette Score only for numeric rows
        def fmt_silhouette(val):
            try:
                return f'{float(val):.4f}'
            except (ValueError, TypeError):
                return val

        st.dataframe(
            quadro.style.format({'Silhouette Score': fmt_silhouette}),
            use_container_width=True, hide_index=True,
        )

    with col_right:
        st.subheader("Curva — Silhouette Score")
        curva_plot = curva[curva['Silhouette'] != 'N/A'].copy()

        if curva_plot.empty:
            st.warning("Nenhum ponto com Silhouette definido para plotar.")
        else:
            curva_plot['Silhouette'] = curva_plot['Silhouette'].astype(float)
            highlight = params.get('k' if algorithm == 'kmeans' else 'preference')

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=curva_plot['Valor'], y=curva_plot['Silhouette'],
                mode='lines+markers', name='Silhouette Score',
            ))
            if highlight in curva_plot['Valor'].values:
                row = curva_plot[curva_plot['Valor'] == highlight].iloc[0]
                fig.add_trace(go.Scatter(
                    x=[highlight],
                    y=[row['Silhouette']],
                    mode='markers',
                    marker=dict(size=14, color='red', symbol='star'),
                    name='Selecionado',
                ))
            fig.update_layout(
                xaxis_title=param_name,
                yaxis_title='Silhouette Score',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 8.2: Smoke-test the clustering page**

With the app running:
1. Navigate to "Clusterização"
2. Select "KMeans", `k=3`
3. Click "Executar Ensaio"

Expected: quadro with 2 rows (Default, Tunado) showing N° Clusters and Silhouette Score, curve shows Silhouette for k=2..10 with red star at k=3.

---

## Task 9: Full test suite run

- [ ] **Step 9.1: Run all tests**

```bash
cd D:/repos/projetos/ml_trials_algorithm
pytest tests/ -v
```

Expected: all tests PASS, zero failures.

- [ ] **Step 9.2: Run app and navigate all three pages**

```bash
streamlit run app.py
```

Navigate to each page, execute one trial per page, and confirm:
- Classification: quadro 5×5 + curve with 20 points (KNN)
- Regression: quadro 5×5 with MAPE in % + curve (XGBoost, 6 n_estimators points)
- Clustering: quadro 2×3 + Silhouette curve (KMeans, 9 points)
