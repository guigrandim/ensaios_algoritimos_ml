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


def run_sweep(algorithm: str, X) -> dict:
    """Sweep the main hyperparameter — result is algorithm-only cacheable."""
    if algorithm not in ('kmeans', 'affinity_propagation'):
        raise ValueError(f"Unknown algorithm: {algorithm}")

    X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)

    if algorithm == 'kmeans':
        rows = []
        for k_val in _KMEANS_SWEEP:
            m = KMeans(n_clusters=k_val, n_init=10, random_state=42)
            sil = _silhouette(X_arr, m.fit_predict(X_arr))
            if sil != 'N/A':
                rows.append({'Valor': k_val, 'Silhouette': float(sil)})
        return {'curva': pd.DataFrame(rows), 'param_name': 'k'}

    rows = []
    for pref_val in _AP_SWEEP:
        m = AffinityPropagation(preference=pref_val, damping=0.6, max_iter=1000, random_state=42)
        sil = _silhouette(X_arr, m.fit_predict(X_arr))
        rows.append({'Valor': pref_val, 'Silhouette': sil})
    return {'curva': pd.DataFrame(rows), 'param_name': 'preference'}


def run_quadro(algorithm: str, params: dict, X) -> dict:
    """Fit default + tuned models and return the comparison table."""
    if algorithm not in ('kmeans', 'affinity_propagation'):
        raise ValueError(f"Unknown algorithm: {algorithm}")

    X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)

    if algorithm == 'kmeans':
        m_def = KMeans(n_init=10, random_state=42)
        labels_def = m_def.fit_predict(X_arr)
        n_def   = len(set(labels_def))
        sil_def = _silhouette(X_arr, labels_def)

        k = params.get('k', 3)
        m_tuned = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels_tuned = m_tuned.fit_predict(X_arr)
        sil_tuned = _silhouette(X_arr, labels_tuned)

        return {'quadro': pd.DataFrame([
            {'Configuração': 'Default', 'N° Clusters': n_def, 'Silhouette Score': sil_def},
            {'Configuração': 'Tunado',  'N° Clusters': k,     'Silhouette Score': sil_tuned},
        ])}

    m_def = AffinityPropagation(random_state=42)
    labels_def = m_def.fit_predict(X_arr)
    n_def   = len(m_def.cluster_centers_indices_)
    sil_def = _silhouette(X_arr, labels_def)

    pref = params.get('preference', -100)
    m_tuned = AffinityPropagation(preference=pref, damping=0.6, max_iter=1000, random_state=42)
    labels_tuned = m_tuned.fit_predict(X_arr)
    n_tuned   = len(m_tuned.cluster_centers_indices_)
    sil_tuned = _silhouette(X_arr, labels_tuned)

    return {'quadro': pd.DataFrame([
        {'Configuração': 'Default', 'N° Clusters': n_def,   'Silhouette Score': sil_def},
        {'Configuração': 'Tunado',  'N° Clusters': n_tuned, 'Silhouette Score': sil_tuned},
    ])}


def run_trial(algorithm: str, params: dict, X) -> dict:
    if algorithm not in ('kmeans', 'affinity_propagation'):
        raise ValueError(f"Unknown algorithm: {algorithm}")
    sweep = run_sweep(algorithm, X)
    quadro_result = run_quadro(algorithm, params, X)
    return {**quadro_result, **sweep}
