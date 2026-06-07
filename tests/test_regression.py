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


def test_ridge_structure(reg_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = reg_data
    result = run_trial('ridge', {'alpha': 1.0}, X_tr, y_tr, X_v, y_v, X_te, y_te)
    assert result['quadro'].shape == (5, 5)
    assert result['param_name'] == 'alpha'
    assert len(result['curva']) == 7


def test_elasticnet_structure(reg_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = reg_data
    result = run_trial('elasticnet', {'alpha': 0.1, 'l1_ratio': 0.5},
                       X_tr, y_tr, X_v, y_v, X_te, y_te)
    assert result['quadro'].shape == (5, 5)
    assert result['param_name'] == 'alpha'
    assert len(result['curva']) == 7


def test_random_forest_structure(reg_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = reg_data
    result = run_trial('random_forest', {'max_depth': 5, 'n_estimators': 50},
                       X_tr, y_tr, X_v, y_v, X_te, y_te)
    assert result['quadro'].shape == (5, 5)
    assert result['param_name'] == 'max_depth'
    assert len(result['curva']) == 20


def test_lightgbm_structure(reg_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = reg_data
    result = run_trial('lightgbm', {'max_depth': -1, 'n_estimators': 100,
                                    'learning_rate': 0.1, 'num_leaves': 31,
                                    'min_child_samples': 20},
                       X_tr, y_tr, X_v, y_v, X_te, y_te)
    assert result['quadro'].shape == (5, 5)
    assert result['param_name'] == 'n_estimators'
    assert len(result['curva']) == 6


def test_polynomial_lasso_structure(reg_data):
    X_tr, y_tr, X_v, y_v, X_te, y_te = reg_data
    result = run_trial('polynomial_lasso', {'degree': 2, 'alpha': 0.01},
                       X_tr, y_tr, X_v, y_v, X_te, y_te)
    assert result['quadro'].shape == (5, 5)
    assert result['param_name'] == 'degree'
    assert len(result['curva']) == 5
