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


def _resolve_params(algorithm: str, params: dict) -> dict:
    merged = dict(_DEFAULTS.get(algorithm, {}))
    merged.update(params)
    return merged


def _build_model(algorithm: str, params: dict):
    if algorithm == 'linear_regression':
        return LinearRegression()
    if algorithm == 'lasso':
        return Lasso(alpha=params['alpha'], max_iter=5000)
    if algorithm == 'ridge':
        return Ridge(alpha=params['alpha'])
    if algorithm == 'elasticnet':
        return ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio'], max_iter=5000)
    if algorithm == 'polynomial':
        return Pipeline([
            ('poly', PolynomialFeatures(degree=params['degree'])),
            ('model', LinearRegression()),
        ])
    if algorithm == 'polynomial_lasso':
        return Pipeline([
            ('poly', PolynomialFeatures(degree=params['degree'])),
            ('model', Lasso(alpha=params['alpha'], max_iter=5000)),
        ])
    if algorithm == 'polynomial_ridge':
        return Pipeline([
            ('poly', PolynomialFeatures(degree=params['degree'])),
            ('model', Ridge(alpha=params['alpha'])),
        ])
    if algorithm == 'polynomial_elasticnet':
        return Pipeline([
            ('poly', PolynomialFeatures(degree=params['degree'])),
            ('model', ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio'], max_iter=5000)),
        ])
    if algorithm == 'decision_tree':
        return DecisionTreeRegressor(max_depth=params.get('max_depth'), random_state=42)
    if algorithm == 'random_forest':
        return RandomForestRegressor(
            max_depth=params.get('max_depth'),
            n_estimators=params['n_estimators'],
            random_state=42, n_jobs=-1,
        )
    if algorithm == 'xgboost':
        return XGBRegressor(
            max_depth=params['max_depth'],
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            random_state=42, verbosity=0,
        )
    if algorithm == 'lightgbm':
        return LGBMRegressor(
            max_depth=params['max_depth'],
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            num_leaves=params['num_leaves'],
            min_child_samples=params['min_child_samples'],
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


def run_sweep(algorithm: str, X_train, y_train, X_val, y_val) -> dict:
    """Sweep the main hyperparameter using defaults — result is algorithm-only cacheable."""
    if algorithm not in _PARAM_NAMES:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    param_name = _PARAM_NAMES[algorithm]
    if param_name is None:
        return {'curva': None, 'param_name': None}

    def ravel(y):
        return y.values.ravel() if hasattr(y, 'values') else np.array(y).ravel()

    y_tr, y_v = ravel(y_train), ravel(y_val)
    rows = []
    for val in _SWEEP_RANGES[algorithm]:
        sp = _resolve_params(algorithm, {})
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
    return {'curva': pd.DataFrame(rows), 'param_name': param_name}


def run_quadro(algorithm: str, params: dict,
               X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """Fit 3 models (default, tuned, final) and return the comparison table."""
    if algorithm not in _PARAM_NAMES:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    def ravel(y):
        return y.values.ravel() if hasattr(y, 'values') else np.array(y).ravel()

    y_tr, y_v, y_te = ravel(y_train), ravel(y_val), ravel(y_test)

    m_def = _build_model(algorithm, _resolve_params(algorithm, {}))
    m_def.fit(X_train, y_tr)
    m_train_def = _reg_metrics(y_tr, m_def.predict(X_train))
    m_val_def   = _reg_metrics(y_v,  m_def.predict(X_val))

    m_t = _build_model(algorithm, _resolve_params(algorithm, params))
    m_t.fit(X_train, y_tr)
    m_train_t = _reg_metrics(y_tr, m_t.predict(X_train))
    m_val_t   = _reg_metrics(y_v,  m_t.predict(X_val))

    X_final = (pd.concat([X_train, X_val])
               if isinstance(X_train, pd.DataFrame)
               else np.vstack([X_train, X_val]))
    y_final = np.concatenate([y_tr, y_v])
    m_final = _build_model(algorithm, _resolve_params(algorithm, params))
    m_final.fit(X_final, y_final)
    m_test = _reg_metrics(y_te, m_final.predict(X_test))

    return {'quadro': pd.DataFrame([
        {'Conjunto': 'Treino Default',    **m_train_def},
        {'Conjunto': 'Validação Default', **m_val_def},
        {'Conjunto': 'Treino Tunado',     **m_train_t},
        {'Conjunto': 'Validação Tunada',  **m_val_t},
        {'Conjunto': 'Teste Final',       **m_test},
    ])}


def run_trial(algorithm: str, params: dict,
              X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    if algorithm not in _PARAM_NAMES:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    sweep = run_sweep(algorithm, X_train, y_train, X_val, y_val)
    quadro_result = run_quadro(algorithm, params, X_train, y_train, X_val, y_val, X_test, y_test)
    return {**quadro_result, **sweep}
