from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from experiments.classification import run_sweep, run_quadro

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

# Clear quadro when algorithm changes
if st.session_state.get('clf_algo') != algorithm:
    st.session_state.pop('clf_quadro', None)
    st.session_state['clf_algo'] = algorithm

# Sweep is cached per algorithm — fast after first run
sweep = cached_sweep(algorithm, X_train, y_train, X_val, y_val)
curva      = sweep['curva']
param_name = sweep['param_name']

# Button only runs the 3-model quadro (fast)
if run:
    with st.spinner("Calculando métricas..."):
        result = run_quadro(algorithm, params, X_train, y_train, X_val, y_val, X_test, y_test)
    st.session_state['clf_quadro'] = result['quadro']

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Quadro Comparativo")
    if 'clf_quadro' in st.session_state:
        fmt = {c: '{:.4f}' for c in ['Accuracy', 'Precision', 'Recall', 'F1-Score']}
        st.dataframe(
            st.session_state['clf_quadro'].style.format(fmt),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Clique em 'Executar Ensaio' para calcular as métricas.")

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
