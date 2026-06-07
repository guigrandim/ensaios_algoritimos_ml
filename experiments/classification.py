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
        'Precision': round(float(mt.precision_score(y_true, y_pred, average='binary', zero_division=0)), 4),
        'Recall':    round(float(mt.recall_score(y_true, y_pred, average='binary', zero_division=0)), 4),
        'F1-Score':  round(float(mt.f1_score(y_true, y_pred, average='binary', zero_division=0)), 4),
    }


def run_sweep(algorithm: str, X_train, y_train, X_val, y_val) -> dict:
    """Sweep the main hyperparameter using defaults — result is algorithm-only cacheable."""
    if algorithm not in _PARAM_NAMES:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    def ravel(y):
        return y.values.ravel() if hasattr(y, 'values') else np.array(y).ravel()

    y_tr, y_v = ravel(y_train), ravel(y_val)
    param_name = _PARAM_NAMES[algorithm]
    rows = []
    for val in _SWEEP_RANGES[algorithm]:
        sp = dict(_DEFAULTS[algorithm])
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
    return {'curva': pd.DataFrame(rows), 'param_name': param_name}


def run_quadro(algorithm: str, params: dict,
               X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """Fit 3 models (default, tuned, final) and return the comparison table."""
    if algorithm not in _PARAM_NAMES:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    def ravel(y):
        return y.values.ravel() if hasattr(y, 'values') else np.array(y).ravel()

    y_tr, y_v, y_te = ravel(y_train), ravel(y_val), ravel(y_test)

    m_def = _build_model(algorithm, _DEFAULTS[algorithm])
    m_def.fit(X_train, y_tr)
    m_train_def = _clf_metrics(y_tr, m_def.predict(X_train))
    m_val_def   = _clf_metrics(y_v,  m_def.predict(X_val))

    m_t = _build_model(algorithm, params)
    m_t.fit(X_train, y_tr)
    m_train_t = _clf_metrics(y_tr, m_t.predict(X_train))
    m_val_t   = _clf_metrics(y_v,  m_t.predict(X_val))

    X_final = (pd.concat([X_train, X_val])
               if isinstance(X_train, pd.DataFrame)
               else np.vstack([X_train, X_val]))
    y_final = np.concatenate([y_tr, y_v])
    m_final = _build_model(algorithm, params)
    m_final.fit(X_final, y_final)
    m_test = _clf_metrics(y_te, m_final.predict(X_test))

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
