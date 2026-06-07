# Design: Streamlit Trial Explorer

**Data:** 2026-06-07
**Projeto:** ml_trials_algorithm
**Objetivo:** App Streamlit interativo para explorar e re-executar os ensaios de ML por categoria, com controle de hiperparâmetros, quadro comparativo de métricas e curva de performance.

---

## 1. Problema

O projeto contém 18 notebooks com ensaios experimentais de algoritmos de ML em três categorias (Classificação, Regressão, Clusterização). Os resultados estão dispersos nos notebooks e no `resultados_ensaio.xlsx`. Não existe interface interativa para explorar como a performance muda com os hiperparâmetros em tempo real.

---

## 2. Solução

App Streamlit com uma página por categoria. O usuário seleciona o algoritmo, ajusta os hiperparâmetros via sliders e clica "Executar Ensaio". O app re-executa o modelo na hora, exibe o quadro comparativo (treino/validação/teste) e a curva de performance por hiperparâmetro.

---

## 3. Arquitetura

### Estrutura de arquivos

```
ml_trials_algorithm/
├── app.py                          # home page (descrição do projeto + navegação)
├── pages/
│   ├── 1_Classificacao.py          # página de classificação
│   ├── 2_Regressao.py              # página de regressão
│   └── 3_Clusterizacao.py          # página de clusterização
└── experiments/
    ├── __init__.py
    ├── classification.py           # lógica ML para classificação
    ├── regression.py               # lógica ML para regressão
    └── clustering.py               # lógica ML para clusterização
```

### Separação de responsabilidades

- **`experiments/`**: funções puras. Recebem DataFrames e parâmetros, devolvem dicionários com métricas. Sem dependência de Streamlit — testáveis isoladamente.
- **`pages/`**: responsável pela UI. Coleta parâmetros do sidebar, chama `experiments/`, renderiza resultados.
- **`app.py`**: apenas a home page com descrição e instrução de navegação.

---

## 4. Fluxo de dados

```
Usuário ajusta sliders → clica "Executar Ensaio"
        ↓
pages/X_Categoria.py coleta parâmetros do sidebar
        ↓
chama experiments/categoria.run_trial(algorithm, params)
        ↓
experiments/ carrega CSVs via @st.cache_data (uma vez por sessão)
        ↓
treina com params default       → métricas treino/val
treina com params tunados       → métricas treino/val
une treino+val, retreina        → métricas teste
varre range do hiperparâmetro  → série para curva de performance
        ↓
retorna { quadro_comparativo: DataFrame, curva: DataFrame }
        ↓
página renderiza tabela + gráfico Plotly
```

---

## 5. UI por página

### 5.1 Classificação (`pages/1_Classificacao.py`)

**Sidebar:**
- Selectbox: `KNN | Decision Tree | Random Forest | Logistic Regression`
- Sliders por algoritmo:
  - KNN: `k` (1–20)
  - Decision Tree: `max_depth` (1–20, mais opção None)
  - Random Forest: `max_depth` (1–20) + `n_estimators` (50–300, step 50)
  - Logistic Regression: `C` (0.001, 0.01, 0.1, 1.0, 10.0, 100.0 — selectbox)
- Botão "Executar Ensaio"

**Área principal:**
- Quadro comparativo: 5 linhas × 4 colunas (Accuracy, Precision, Recall, F1-Score)
  - Linhas: Treino Default | Validação Default | Treino Tunado | Validação Tunada | Teste Final
- Selectbox de métrica acima do gráfico (Accuracy / Precision / Recall / F1-Score)
- Gráfico de linha (Plotly): hiperparâmetro principal no eixo X, métrica no Y, duas linhas (Treino, Validação), ponto selecionado destacado

### 5.2 Regressão (`pages/2_Regressao.py`)

**Sidebar:**
- Selectbox: `Linear Regression | Lasso | Ridge | ElasticNet | Polynomial | Polynomial Lasso | Polynomial Ridge | Polynomial ElasticNet | Decision Tree | Random Forest | XGBoost | LightGBM`
- Sliders por algoritmo:
  - Linear Regression: sem hiperparâmetros (nota informativa no sidebar)
  - Lasso / Ridge / ElasticNet: `alpha` (selectbox: 0.001, 0.01, 0.1, 1.0, 10.0)
  - Polynomial (×4): `degree` (1–4)
  - Decision Tree: `max_depth` (1–20)
  - Random Forest: `max_depth` (1–20) + `n_estimators` (50–300, step 50)
  - XGBoost / LightGBM: `max_depth` (3–10) + `n_estimators` (100–600, step 100) + `learning_rate` (selectbox: 0.01, 0.05, 0.1, 0.3)
- Botão "Executar Ensaio"

**Área principal:**
- Quadro comparativo: 5 linhas × 4 colunas (R², RMSE, MAE, MAPE)
- Selectbox de métrica (R² / RMSE / MAE / MAPE)
- Gráfico de linha: mesmo padrão da classificação

### 5.3 Clusterização (`pages/3_Clusterizacao.py`)

**Sidebar:**
- Selectbox: `KMeans | Affinity Propagation`
- Sliders por algoritmo:
  - KMeans: `k` (2–10)
  - Affinity Propagation: `preference` (selectbox: -50, -100, -200, -300, -500, -700, -1000)
- Botão "Executar Ensaio"

**Área principal:**
- Quadro comparativo: 2 linhas (Default, Tunado) × 2 colunas (N° Clusters, Silhouette Score)
- Gráfico de linha: Silhouette Score × valor do parâmetro varrido (range fixo, igual ao dos notebooks)

---

## 6. Cache e performance

- Datasets carregados com `@st.cache_data` — lidos do disco uma única vez por sessão.
- A curva de performance varre um range fixo pré-definido a cada execução (não depende do slider). O slider define qual ponto é destacado no gráfico.
- Não há cache de resultados de modelos — cada clique em "Executar Ensaio" re-executa do zero. Isso é aceitável dado o tamanho dos datasets do projeto.

---

## 7. Tratamento de erros

| Cenário | Comportamento |
|---------|---------------|
| Dataset não encontrado no path esperado | `st.error()` com path esperado, sem stack trace |
| Silhouette Score indefinido (1 cluster) | Célula exibe `"N/A"`, ponto omitido do gráfico |
| Affinity Propagation sem convergência | `st.warning()` descritivo, execução interrompida graciosamente |
| Algoritmo sem hiperparâmetros (Linear Regression) | Sidebar exibe nota; curva de performance omitida; só quadro comparativo |

---

## 8. Dependências adicionais

O projeto já usa `scikit-learn`, `pandas`, `numpy`, `xgboost`, `lightgbm`. Será necessário adicionar:
- `streamlit`
- `plotly`

---

## 9. Fora do escopo

- Comparação cross-categoria numa mesma tela
- Upload de dataset customizado pelo usuário
- Persistência de resultados entre sessões
- Deploy em cloud (apenas execução local)
