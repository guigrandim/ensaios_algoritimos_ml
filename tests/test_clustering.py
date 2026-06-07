import pytest
from experiments.clustering import run_trial
from experiments.clustering import _silhouette
import numpy as np


def test_kmeans_structure(cluster_data):
    result = run_trial('kmeans', {'k': 3}, cluster_data)
    assert set(result.keys()) == {'quadro', 'curva', 'param_name'}
    assert result['quadro'].shape == (2, 3)
    assert list(result['quadro']['Configuração']) == ['Default', 'Tunado']
    assert set(result['quadro'].columns) == {'Configuração', 'N° Clusters', 'Silhouette Score'}
    assert result['param_name'] == 'k'
    assert 1 <= len(result['curva']) <= 9  # sweep k=2..10, some may be dropped if N/A


def test_affinity_propagation_structure(cluster_data):
    result = run_trial('affinity_propagation', {'preference': -100}, cluster_data)
    assert result['quadro'].shape == (2, 3)
    assert result['param_name'] == 'preference'
    assert len(result['curva']) == 7  # 7 preference values


def test_silhouette_returns_na_for_single_cluster():
    X = np.random.randn(50, 2)
    labels_all_same = np.zeros(50, dtype=int)  # all points in cluster 0
    assert _silhouette(X, labels_all_same) == 'N/A'


def test_silhouette_returns_float_for_valid_clusters():
    X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    result = _silhouette(X, labels)
    assert isinstance(result, float)
    assert 0 < result <= 1


def test_kmeans_curva_has_silhouette_column(cluster_data):
    result = run_trial('kmeans', {'k': 3}, cluster_data)
    assert 'Silhouette' in result['curva'].columns
    assert 'Valor' in result['curva'].columns


def test_unknown_algorithm_raises(cluster_data):
    with pytest.raises(ValueError, match="Unknown algorithm"):
        run_trial('dbscan', {}, cluster_data)
