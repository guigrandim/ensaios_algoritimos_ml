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
