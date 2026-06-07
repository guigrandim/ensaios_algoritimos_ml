from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from experiments.regression import run_sweep, run_quadro

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


@st.cache_data
def cached_sweep(algorithm, X_train, y_train, X_val, y_val):
    return run_sweep(algorithm, X_train, y_train, X_val, y_val)


try:
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
except FileNotFoundError as e:
    st.error(f"Dataset não encontrado: {e}")
    st.stop()

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
            params['num_leaves']        = st.slider('num_leaves', 10, 100, 31)
            params['min_child_samples'] = st.slider('min_child_samples', 5, 50, 20)

    run = st.button("Executar Ensaio", type="primary", use_container_width=True)

# Clear quadro when algorithm changes
if st.session_state.get('reg_algo') != algorithm:
    st.session_state.pop('reg_quadro', None)
    st.session_state['reg_algo'] = algorithm

# Sweep cached per algorithm — fast after first run
sweep = cached_sweep(algorithm, X_train, y_train, X_val, y_val)
curva      = sweep['curva']
param_name = sweep['param_name']

# Button only runs the 3-model quadro (fast)
if run:
    with st.spinner("Calculando métricas..."):
        result = run_quadro(algorithm, params, X_train, y_train, X_val, y_val, X_test, y_test)
    st.session_state['reg_quadro'] = result['quadro']

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Quadro Comparativo")
    if 'reg_quadro' in st.session_state:
        fmt = {'R²': '{:.4f}', 'RMSE': '{:.4f}', 'MAE': '{:.4f}', 'MAPE': '{:.2%}'}
        st.dataframe(
            st.session_state['reg_quadro'].style.format(fmt),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Clique em 'Executar Ensaio' para calcular as métricas.")

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
