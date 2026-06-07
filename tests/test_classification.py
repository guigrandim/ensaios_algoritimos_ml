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


def test_curve_values_in_range(clf_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = clf_data
    result = run_trial('decision_tree', {'max_depth': 5}, X_tr, y_tr, X_v, y_v, X_te, y_te)
    curva = result['curva']
    for col in ['Acc_train', 'Acc_val', 'F1_train', 'F1_val']:
        assert curva[col].between(0.0, 1.0).all(), f"curva['{col}'] out of [0, 1]"
    assert curva['Valor'].notna().all()
