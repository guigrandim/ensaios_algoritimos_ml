from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from experiments.clustering import run_sweep, run_quadro

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


@st.cache_data
def cached_sweep(algorithm, X):
    return run_sweep(algorithm, X)


try:
    X = load_data()
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
    if algorithm == 'kmeans':
        params['k'] = st.slider('k (clusters)', 2, 10, 3)
    elif algorithm == 'affinity_propagation':
        params['preference'] = st.select_slider(
            'preference', options=[-50, -100, -200, -300, -500, -700, -1000], value=-100)
    run = st.button("Executar Ensaio", type="primary", use_container_width=True)

# Clear quadro when algorithm changes
if st.session_state.get('clt_algo') != algorithm:
    st.session_state.pop('clt_quadro', None)
    st.session_state['clt_algo'] = algorithm

# Sweep cached per algorithm — fast after first run
sweep = cached_sweep(algorithm, X)
curva      = sweep['curva']
param_name = sweep['param_name']

# Button only runs the 2-model quadro (fast)
if run:
    with st.spinner("Calculando métricas..."):
        result = run_quadro(algorithm, params, X)
    st.session_state['clt_quadro'] = result['quadro']

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Quadro Comparativo")
    if 'clt_quadro' in st.session_state:
        def fmt_silhouette(val):
            try:
                return f'{float(val):.4f}'
            except (ValueError, TypeError):
                return val

        st.dataframe(
            st.session_state['clt_quadro'].style.format({'Silhouette Score': fmt_silhouette}),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("Clique em 'Executar Ensaio' para calcular as métricas.")

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
        if highlight is not None and highlight in curva_plot['Valor'].values:
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
