# Detecção de Fraudes em Transações com Cartão de Crédito

Projeto desenvolvido como parte do Bootcamp de dados, explorando técnicas de
machine learning para identificar transações fraudulentas em um dataset altamente
desbalanceado.

## Dataset

- Fonte: [creditcard.csv](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv) — disponibilizado pelo TensorFlow
- 284.807 transações, sendo apenas 0,17% fraudes (Classe 1)
- Features V1 a V28 são componentes PCA anonimizadas; demais colunas: `Time`, `Amount` e `Class`

## Etapas do Projeto

### 1. Importação e Visualização
Carregamento do dataset via URL com pandas e análise inicial das distribuições.

### 2. Análise do Desbalanceamento
Identificação do desequilíbrio severo entre classes:
- Classe 0 (legítima): 99,83%
- Classe 1 (fraude): 0,17%

### 3. Feature Engineering
- Transformação logarítmica do valor da transação (`Amount_log`)
- Normalização com `StandardScaler`

### 4. Modelos Treinados

| Modelo | F1 (fraude) | AUC |
|---|---|---|
| Logistic Regression | 0.73 | 0.927 |
| Random Forest (balanced) | 0.79 | — |
| Pipeline + threshold 0.3 | 0.72 | — |

### 5. Técnicas de Balanceamento
- **Undersampling**: redução da classe majoritária
- **Oversampling com SMOTE**: geração sintética de amostras da classe minoritária

### 6. Ajuste de Threshold
Limiar de decisão ajustado para 0.3 (padrão 0.5) para aumentar a sensibilidade
na detecção de fraudes, priorizando recall sobre precisão.

### 7. Avaliação
- Matriz de confusão
- Curva ROC e AUC
- Curva Precision-Recall
- Classification Report

## Tecnologias Utilizadas

- Python 3.12
- pandas, numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib

## Como Executar

Abra diretamente no Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1gb9HlYDT_Ki0oKNqiU26UX7kSuFGhd-M?usp=sharing)

## Principais Aprendizados

- Datasets desbalanceados exigem métricas além da acurácia (F1, AUC, Recall)
- O Random Forest com `class_weight="balanced"` superou a Regressão Logística
na detecção de fraudes
- Ajustar o threshold de decisão é uma estratégia eficaz quando o custo de
falsos negativos é alto
