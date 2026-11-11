# Credit Card Fraud Detection

A notebook comparison of three approaches for detecting fraudulent credit card transactions. The goal is to compare precision, recall, F1, and AUPRC on the same held-out test set.

## What the notebook does

- Loads `creditcard.csv` and checks the class distribution.
- Standardizes transaction `Amount` and `Time` features; PCA features are already scaled.
- Creates stratified fit, validation, and test partitions. SMOTE is applied only to model-fit data for the SMOTE-based methods.
- Trains three models and selects each model's decision threshold on validation data by maximizing F1.
- Reports test-set confusion matrices, precision, recall, F1, and AUPRC, with comparison plots.

## Algorithms compared

| Model | Imbalance strategy | Model configuration |
|---|---|---|
| XGBoost | Trains on original imbalanced fit data; uses `scale_pos_weight` | 300 estimators, max depth 6, learning rate 0.05 |
| XGBoost without `scale_pos_weight` | Trains on SMOTE-resampled fit data | 300 estimators, max depth 6, learning rate 0.05 |
| Random Forest | Trains on SMOTE-resampled fit data | 300 trees, max depth 10 |

## Results

Results from the included run on `creditcard.csv` (284,807 transactions; 492 fraud cases). Thresholds were selected using the validation partition; all metrics below are from the held-out test partition.

| Model | Threshold | Precision | Recall | F1 | AUPRC |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.9392 | **0.9512** | 0.7959 | **0.8667** | **0.8756** |
| XGBoost without `scale_pos_weight` | 0.9107 | 0.8265 | **0.8265** | 0.8265 | 0.8559 |
| Random Forest | 0.7388 | 0.7451 | 0.7755 | 0.7600 | 0.7939 |

| Model | True negatives | False positives | False negatives | True positives |
|---|---:|---:|---:|---:|
| XGBoost | 56,860 | 4 | 20 | 78 |
| XGBoost without `scale_pos_weight` | 56,847 | 17 | 17 | 81 |
| Random Forest | 56,838 | 26 | 22 | 76 |

Weighted XGBoost achieved the highest precision, F1, and AUPRC in this run. SMOTE-based XGBoost detected the most fraud cases (highest recall), with more false positives. These are results from one stratified split; model hyperparameters were not tuned.

## Setup and run

1. Put `creditcard.csv` in the project root. The dataset is available from the [Kaggle credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Open and run `credit_card_fraud_model_comparison.ipynb` from top to bottom.

## Files

- `credit_card_fraud_model_comparison.ipynb` — data preparation, model training, threshold selection, evaluation, and plots.
- `requirements.txt` — pinned Python dependencies.
