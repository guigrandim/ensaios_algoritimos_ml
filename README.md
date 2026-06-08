# Ensaios de Algoritmos de Machine Learning

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Repositório de ensaios comparativos de algoritmos de machine learning organizados em três categorias: **Classificação**, **Regressão** e **Clusterização**.

<p align="center">
<img src="./assets/img/fluxo.png" alt="Ensaios de Algoritmos de Machine Learning" width="800px">
</p>

## 1. Problema de Negócio

**Contexto:**
- A empresa Data Money atua com consultorias em Ciência de Dados
- Acredita que a **expertise no treinamento e ajuste fino dos algoritmos** é o diferencial competitivo que entrega ótimos resultados aos clientes

**Objetivo:**
- Realizar **ensaios experimentais** com algoritmos de:
  - **Classificação**
  - **Regressão**
  - **Clusterização**
- Estudar como a **performance muda** à medida que os principais **parâmetros de controle de overfitting e underfitting** são alterados

**O que se busca entender:**
- Como cada algoritmo se comporta em diferentes configurações de parâmetros
- O ponto de equilíbrio entre **underfitting** (modelo simples demais) e **overfitting** (modelo complexo demais)
- Quais parâmetros têm maior impacto na generalização do modelo
- Como replicar o conhecimento adquirido para futuras consultorias

**Valor gerado:**
- Documentar empiricamente o comportamento dos algoritmos
- Criar um **guia interno** de boas práticas para tuning de modelos
- Fundamentar a expertise da equipe com evidências práticas e reproduzíveis

---

## 2. Planejamento da Solução

**Produto Final:** 7 tabelas mostrando a performance dos algoritmos, avaliados usando múltiplas métricas, para 3 conjuntos de dados diferentes: Treinamento, Validação e Teste.

### Algoritmos Ensaiados

| Categoria | Algoritmos | Métricas de Performance |
|-----------|-----------|------------------------|
| **Classificação** | KNN, Decision Tree, Random Forest, Logistic Regression | Accuracy, Precision, Recall, F1-Score |
| **Regressão** | Linear Regression (Lasso, Ridge, ElasticNet), Polynomial (Lasso, Ridge, ElasticNet), Decision Tree, Random Forest, XGBoost, LightGBM | R², MSE, RMSE, MAE, MAPE |
| **Clusterização** | K-Means, Affinity Propagation | Silhouette Score |

### Ferramentas Utilizadas

- 🐍 Python 3.10+
- 📦 Scikit-learn, XGBoost, LightGBM
- 📓 Jupyter Notebooks

---

## 3. Dados

### Classificação — Airline Passenger Satisfaction

Dataset público com avaliações de passageiros de companhia aérea, incluindo características do voo e notas de satisfação por serviço.

| Atributo | Valor |
|----------|-------|
| **Amostras totais** | 129.487 (treino: 72.515 / validação: 31.079 / teste: 25.893) |
| **Features** | 24 (dados demográficos, características do voo e avaliações de serviço a bordo) |
| **Target** | `satisfaction` — binário: 0 = insatisfeito, 1 = satisfeito |
| **Tipo de problema** | Classificação binária |

<details>
<summary>Features disponíveis</summary>

`customer_type`, `age`, `class`, `flight_distance`, `inflight_wifi_service`, `departure_arrival_time_convenient`, `ease_of_online_booking`, `gate_location`, `food_and_drink`, `online_boarding`, `seat_comfort`, `inflight_entertainment`, `on_board_service`, `leg_room_service`, `baggage_handling`, `checkin_service`, `inflight_service`, `cleanliness`, `departure_delay_in_minutes`, `arrival_delay_in_minutes`, `gender_Female`, `gender_Male`, `type_of_travel_business_travel`, `type_of_travel_personal_travel`

</details>

---

### Regressão — Spotify Song Popularity

Dataset com atributos de áudio extraídos pela API do Spotify, com o objetivo de prever a popularidade de músicas.

| Atributo | Valor |
|----------|-------|
| **Amostras totais** | 18.835 (treino: 10.547 / validação: 4.521 / teste: 3.767) |
| **Features** | 13 (atributos de áudio extraídos da API do Spotify) |
| **Target** | `song_popularity` — contínuo, escala 0–100 |
| **Tipo de problema** | Regressão |

<details>
<summary>Features disponíveis</summary>

`song_duration_ms`, `acousticness`, `danceability`, `energy`, `instrumentalness`, `key`, `liveness`, `loudness`, `audio_mode`, `speechiness`, `tempo`, `time_signature`, `audio_valence`

</details>

---

### Clusterização — Wine Dataset

Dataset clássico com medições químicas de amostras de vinho, sem rótulos, usado para descoberta de agrupamentos naturais.

| Atributo | Valor |
|----------|-------|
| **Amostras totais** | 178 (sem split — aprendizado não supervisionado) |
| **Features** | 13 (propriedades químicas do vinho) |
| **Target** | Nenhum (não supervisionado) |
| **Tipo de problema** | Clusterização |

<details>
<summary>Features disponíveis</summary>

`alcohol`, `malic_acid`, `ash`, `ash_alcanity`, `magnesium`, `total_phenols`, `flavanoids`, `nonflavanoid_phenols`, `proanthocyanins`, `color_intensity`, `hue`, `od280`, `proline`

</details>

---

## 4. Desenvolvimento

**Estratégia da Solução:** Para ensaiar os algoritmos de Machine Learning, foram escritos códigos em Python para treinar cada um dos algoritmos e realizar variações dos principais parâmetros de cada algoritmo para ajuste de overfitting e underfitting, observando a métrica final.

### Passo a Passo

| Passo | Descrição |
|-------|-----------|
| **Passo 1** | Treinar os algoritmos com os dados de treinamento utilizando os parâmetros default |
| **Passo 2** | Medir a performance no conjunto de treinamento (parâmetros default) |
| **Passo 3** | Medir a performance no conjunto de validação (parâmetros default) |
| **Passo 4** | Ajustar os hiperparâmetros para encontrar o conjunto que minimize overfitting/underfitting |
| **Passo 5** | Unir os dados de treinamento e validação |
| **Passo 6** | Retreinar o algoritmo com a união dos dados e os melhores parâmetros encontrados |
| **Passo 7** | Medir a performance final no conjunto de teste |
| **Passo 8** | Quadro Comparativo — análise diagnóstica completa do ensaio |

---

## 5. Estrutura do Projeto (Notebooks)

```
ml_trials_algorithm/
├── notebooks/
│   ├── classificacao/       # KNN, Decision Tree, Logistic Regression, Random Forest
│   ├── regressao/           # Linear, Lasso, Ridge, ElasticNet, Polynomial, Decision Tree, Random Forest, XGBoost, LightGBM
│   └── clusterizacao/       # KMeans, Affinity Propagation
└── dataset/
    ├── classification_datasets/
    │   ├── a_traninig/      # X_training.csv, y_training.csv
    │   ├── b_validation/    # X_validation.csv, y_validation.csv
    │   └── c_test/          # X_test.csv, y_test.csv
    ├── regression_datasets/
    │   ├── a_traninig/
    │   ├── b_validation/
    │   └── c_test/
    └── clusters_datasets/
        └── a_traning/       # X_dataset.csv (sem split — unsupervised)
```

### Notebooks — Classificação

| Notebook | Abrir |
|----------|-------|
| KNN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/classificacao/ensaios_ml_knn.ipynb) |
| Decision Tree | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/classificacao/ensaios_ml_decisiontree.ipynb) |
| Random Forest | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/classificacao/ensaios_ml_randomforest.ipynb) |
| Logistic Regression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/classificacao/ensaios_ml_logistc_regression.ipynb) |

### Notebooks — Regressão

| Notebook | Abrir |
|----------|-------|
| Linear Regression | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_linear_regression.ipynb) |
| Linear Regression — Lasso | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_linear_regression_lasso.ipynb) |
| Linear Regression — Ridge | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_linear_regression_ridge.ipynb) |
| Linear Regression — ElasticNet | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_linear_regression_elastic_net.ipynb) |
| Polynomial Regressor | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_polynomial_regressor.ipynb) |
| Polynomial Regressor — Lasso | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_polynomial_regressor_lasso.ipynb) |
| Polynomial Regressor — Ridge | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_polynomial_regressor_ridge.ipynb) |
| Polynomial Regressor — ElasticNet | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_polynomial_regressor_elastic_net.ipynb) |
| Decision Tree Regressor | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_decision_tree_regression.ipynb) |
| Random Forest Regressor | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_random_forest_regression.ipynb) |
| XGBoost | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_xgboost.ipynb) |
| LightGBM | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/regressao/ensaios_ml_lightgbm.ipynb) |

### Notebooks — Clusterização

| Notebook | Abrir |
|----------|-------|
| KMeans | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/clusterizacao/ensaios_clustering_kmeans.ipynb) |
| Affinity Propagation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/guigrandim/ensaios_algoritimos_ml/blob/main/notebooks/clusterizacao/ensaios_clustering_ap.ipynb) |

---

## 6. Resultados — Classificação

**Métricas no conjunto de teste (dados nunca vistos durante treino/validação)**

| Algoritmo | Melhores Parâmetros | Acurácia | Precisão | Recall | F1-Score |
|-----------|---------------------|----------|----------|--------|----------|
| **Random Forest** | `max_depth=19`, `n_estimators=100` | **0.9646** | **0.9730** | **0.9456** | **0.9591** |
| **Decision Tree** | `max_depth=14` | 0.9554 | 0.9551 | 0.9427 | 0.9489 |
| **Logistic Regression** | `C=1.0`, `solver=lbfgs`, `max_iter=100` | 0.8711 | 0.8685 | 0.8324 | 0.8501 |
| KNN | `k=4` | 0.6742 | 0.6828 | 0.4815 | 0.5647 |

> Melhor algoritmo: **Random Forest** — maior F1-Score (0.9591) e Precisão (0.9730) no teste.

---

## 7. Resultados — Regressão

**Métricas no conjunto de teste (dados nunca vistos durante treino/validação)**

| Algoritmo | Melhores Parâmetros | R² | RMSE | MAE | MAPE |
|-----------|---------------------|----|------|-----|------|
| **XGBoost** | `n_estimators=500`, `max_depth=7`, `lr=0.1` | **0.3678** | **17.54** | **12.68** | 616.91% |
| **LightGBM** | `n_estimators=500`, `num_leaves=63`, `lr=0.1`, `min_child=10` | 0.3579 | 17.68 | 12.75 | 632.43% |
| Random Forest Regressor | `n_estimators=300`, `max_depth=10` | 0.2462 | 19.16 | 15.26 | 715.39% |
| Polynomial | `degree=2` | 0.0909 | 21.04 | 16.74 | 827.70% |
| Decision Tree Regressor | `max_depth=5` | 0.0896 | 21.05 | 16.83 | 788.62% |
| Polynomial ElasticNet | `degree=2`, `alpha=0.001`, `l1_ratio=0.8` | 0.0886 | 21.07 | 16.76 | 833.11% |
| Polynomial Lasso | `degree=2`, `alpha=0.01` | 0.0854 | 21.10 | 16.79 | 834.21% |
| Polynomial Ridge | `degree=2`, `alpha=1.0` | 0.0854 | 21.10 | 16.79 | 834.21% |
| Linear Regression | sem hiperparâmetros | 0.0512 | 21.49 | 17.14 | 853.14% |
| Lasso | `alpha=0.001` | 0.0511 | 21.49 | 17.14 | 853.31% |
| ElasticNet | `alpha=0.001`, `l1_ratio=0.5` | 0.0511 | 21.49 | 17.14 | 853.71% |
| Ridge | `alpha=10.0` | 0.0511 | 21.49 | 17.14 | 853.78% |

> Melhor algoritmo: **XGBoost** — maior R² (0.3678) e menor RMSE (17.54) e MAE (12.68) no teste.
>
> **Observação:** Os R² baixos indicam que este dataset de regressão possui alta variabilidade não capturada pelos features atuais. Os algoritmos ensemble (XGBoost, LightGBM, Random Forest) se destacam por capturar padrões não-lineares mais complexos.

---

## 8. Resultados — Clusterização

**Avaliação pelo Silhouette Score** (quanto mais próximo de 1, melhor a separação entre clusters)

### KMeans — Grid Search por K

| K | WCSS (Inércia) | Silhouette Score |
|---|----------------|-----------------|
| 2 | 1017.83 | 0.2132 |
| **3** | **829.04** | **0.2330** |
| 4 | 750.15 | 0.2166 |
| 5 | 681.87 | 0.1868 |
| 6 | 622.37 | 0.2203 |
| 7 | 568.97 | 0.2108 |
| 8 | 533.33 | 0.1865 |
| 9 | 505.89 | 0.1867 |
| 10 | 477.39 | 0.1746 |

> Melhor K: **K=3** (Silhouette = 0.2330)

### Affinity Propagation — Variação de Preferência

| Preferência | Clusters Encontrados | Silhouette Score |
|-------------|---------------------|-----------------|
| **-50** | **7** | **0.2023** |
| -100 | 4 | 0.1588 |
| -200 | 3 | 0.1957 |
| -300 | 2 | 0.1762 |
| -500 | 1 | N/A |
| -700 | 1 | N/A |
| -1000 | 1 | N/A |

> Melhor configuração AP: **preference = -50** → 7 clusters, Silhouette = 0.2023
>
> **KMeans supera AP** neste dataset (0.2330 vs 0.2023), além de ser computacionalmente mais eficiente.

---

## 9. Comparativo Final por Categoria

| Categoria | Melhor Algoritmo | Métrica Principal |
|-----------|-----------------|-------------------|
| Classificação | **Random Forest** | F1-Score = 0.9591 |
| Regressão | **XGBoost** | R² = 0.3678 / RMSE = 17.54 |
| Clusterização | **KMeans (K=3)** | Silhouette = 0.2330 |

---

## 10. Conclusões

### Classificação
- **Random Forest** dominou com F1-Score de **0.9591**, seguido de perto pela Decision Tree (0.9489). Ambos os modelos baseados em árvore superaram regressão logística e KNN por larga margem, indicando que os dados possuem **fronteiras de decisão não-lineares** bem capturadas por métodos ensemble.
- 📉 KNN teve desempenho fraco (F1 = 0.5647), possivelmente afetado pela maldição da dimensionalidade em um espaço de features de alta dimensão.

### Regressão
- O desempenho geral foi modesto — **XGBoost** liderou com R² = **0.3678**, mas ainda deixa mais de 63% da variância inexplicada.
- 🔍 Os R² baixos em todos os algoritmos sugerem que as features disponíveis têm **baixo poder preditivo** sobre o target ou que existem relações altamente ruidosas no dataset.
- Algoritmos lineares (Linear Regression, Lasso, Ridge, ElasticNet) ficaram todos agrupados em torno de R² ≈ 0.051, sinalizando que a **linearidade não é suficiente** para capturar os padrões presentes.

### Clusterização
- **KMeans com K=3** obteve o melhor Silhouette Score (0.2330), revelando **3 agrupamentos naturais** no dataset com separação razoável entre clusters.
- Affinity Propagation com `preference=-50` gerou 7 clusters e Silhouette = 0.2023 — próximo do KMeans, mas com estrutura mais fragmentada e custo computacional maior.
- ⚡ KMeans é a escolha recomendada: **mais eficiente** e com resultado ligeiramente superior.

---

## 11. Próximos Passos

### Classificação
- [ ] 🔧 Testar **XGBoost e LightGBM** para classificação — potencial de superar Random Forest com tuning adequado
- [ ] 🔎 Análise de importância de features para identificar variáveis mais relevantes
- [ ] ⚖️ Investigar desbalanceamento de classes — avaliar se SMOTE ou class_weight melhoram o recall

### Regressão
- [ ] 🧪 Explorar **feature engineering** — criação de features derivadas para capturar relações não-lineares
- [ ] 🗑️ Analisar **outliers** no target — o MAPE elevado (~600–850%) sugere presença de valores extremos
- [ ] 📊 Avaliar a qualidade do dataset de regressão — considerar coleta de mais features ou dados adicionais

### Clusterização
- [ ] 🔬 **Interpretar os 3 clusters** do KMeans — analisar médias e distribuição de features por cluster
- [ ] 🌐 Testar outros algoritmos: **DBSCAN** (detecta outliers) e **Hierarchical Clustering** (dendrograma)
- [ ] 📐 Avaliar impacto de **normalização** (StandardScaler) sobre os resultados de clustering
