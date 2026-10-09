# Telecom Customer Churn: Analysis and Modelling

Why do telecom customers leave? This project explores 7,043 customers of a California telecom company,
cleans and encodes the data, trains a gradient-boosted classifier to predict churn, and uses SHAP to
explain which factors drive it.

## Data

`dataset/telecom_customer_churn.csv`: 7,043 customers × 38 columns (demographics, location, services,
contract, charges, revenue, `Customer Status`, churn category and reason). Two derived files in
`dataset/` hold the analysis table and the joined-customer view.

## Notebooks

| Notebook | Content |
|---|---|
| `code/01_data_exploration.ipynb` | First look at the columns, types and missing values |
| `code/02_exploratory_data_analysis.ipynb` | Univariate and churn-vs-feature analysis |
| `code/telecom-customer-churn.ipynb` | Full pipeline: EDA → cleaning → feature engineering → XGBoost → SHAP |

Pipeline highlights:

- **Cleaning:** `Customer Status` becomes the churn label (stayed 0 / churned 1); churn category and
  reason are dropped to avoid leakage; service columns that are empty because the customer has no
  internet are filled with "None" / "No" / 0; the remaining numeric gaps use KNN imputation.
- **Features:** label encoding for yes/no columns, one-hot encoding for the rest; cities are reduced to
  the five largest plus "Other".
- **Model:** XGBoost tuned with `RandomizedSearchCV`; class imbalance handled with `scale_pos_weight`.
- **Explanation:** SHAP `TreeExplainer` summary and dependence plots.

## Results

Hold-out test set of 1,409 customers:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Stayed | 0.93 | 0.88 | 0.91 |
| **Churned** | 0.77 | **0.87** | 0.81 |
| Accuracy | | | **0.87** |

Best parameters: `n_estimators=600, max_depth=5, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8`.

The SHAP analysis points to **tenure**, a **month-to-month contract**, the **number of referrals** and the
**monthly charge** as the strongest churn drivers: new, month-to-month customers with few referrals
and higher bills are the most likely to leave.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap jupyter
jupyter lab                     # open the notebooks in code/
```
