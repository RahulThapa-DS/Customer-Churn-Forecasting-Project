# 📊 Customer Churn Forecasting & Analytics System

An end-to-end production-grade Machine Learning system built to predict, analyze, and mitigate customer churn in telecommunications accounts. This project features automated data preprocessing, domain-specific feature engineering, multi-model benchmarking across seven classification algorithms, hyperparameter tuning, and a serialized inference deployment pipeline.

---

## 📌 Executive Summary
Customer churn presents a significant financial challenge in subscription-based services. Retaining existing customers is considerably cheaper than acquiring new ones. 

This project implements a complete Machine Learning workflow that identifies high-risk churn candidates early, enabling targeted retention campaigns. By benchmarking linear models, tree ensembles, and neural architectures, the finalized pipeline prioritizes **Recall (78%)** and **ROC-AUC (0.8462)** to minimize missed churners while maintaining strong ranking confidence.

### Key Performance Highlights
- **Top Model Architecture:** Logistic Regression (Tuned with `C=10.0`, `solver='lbfgs'`)
- **Final Tuned ROC-AUC Score:** **`0.8462`**
- **Churn Recall (Sensitivity):** **`0.78` (78%)** — successfully captures 78% of all true churn cases.
- **Production Readiness:** Serialized via `joblib` into a standalone pipeline class (`ChurnPredictorPipeline`) supporting real-time risk stratification (`Low Risk`, `Medium Risk`, `High Risk`).

---

## 📁 Repository Structure

```text
telecom-churn-prediction/
│
├── artifacts/                  # Serialized model checkpoints & scalers
│   ├── churn_model.pkl         # Final hyperparameter-tuned model
│   ├── scaler.pkl              # Fitted StandardScaler instance
│   └── feature_names.pkl       # Feature alignment mapping array
│
├── data/                       # Dataset directory
│   ├── raw_dataset.csv         # Initial Telco dataset
│   └── cleaned_dataset.csv     # Preprocessed data snapshot
│
├── notebooks/                  # Interactive Development & Analysis
│   └── churn_prediction.ipynb  # End-to-end workflow notebook
│
├── src/                        # Production Python Source Scripts
│   ├── data_cleaning.py        # Preprocessing & type coercion
│   ├── feature_engineering.py  # Domain ratios & categorical encoding
│   ├── eda_visualizations.py   # Plotting functions (KDE, heatmaps)
│   ├── train_evaluate.py       # Multi-model benchmarking & tuning
│   └── inference_pipeline.py   # Production deployment class
│
├── README.md                   # Technical project documentation
└── requirements.txt            # System dependencies
---
```
## 🛠️ Dataset Profile & Preprocessing

### Dataset Summary
- **Total Customer Accounts:** 7,043 rows
- **Initial Features:** 21 columns (18 categorical, 3 numerical)
- **Target Variable:** `Churn` (`Yes` / `No` $\rightarrow$ mapped to `1` / `0`)
- **Class Distribution:** 73.5% Retained (`0`), 26.5% Churned (`1`)

### Preprocessing & Data Hygiene
- **Identifier Elimination:** Dropped non-predictive attributes (`customerID`, `gender`, `PhoneService`) to eliminate noise and prevent data leakage.
- **Type Coercion & Missing Value Imputation:**
  - `TotalCharges` contained blank space strings (`" "`).
  - Converted blanks to `NaN` using `pd.to_numeric(..., errors='coerce')` and imputed missing values using `MonthlyCharges` (corresponding to new customers with zero tenure).
- **Feature Scaling:** Applied `StandardScaler` to continuous variables (`tenure`, `MonthlyCharges`, `TotalCharges`) to normalize features for distance and gradient-based estimators.
- **Addressing Class Imbalance:** Applied cost-sensitive learning settings natively (`class_weight='balanced'` and `scale_pos_weight`) to adjust loss functions without generating synthetic data.

---

## 🧠 Feature Engineering Strategy

To capture complex customer behaviors, domain-specific continuous and categorical metrics were engineered:

| Feature Name | Mathematical / Logical Formula | Business Rationale |
| :--- | :--- | :--- |
| **`log_tenure`** | $\log(1 + \text{tenure})$ | Normalizes tenure distribution and models diminishing churn risk as tenure grows. |
| **`tenure_cohort`** | `[onboarding, established, loyal, veteran]` | Groups accounts into distinct business lifecycle stages. |
| **`avg_monthly_spend`** | $\frac{\text{TotalCharges}}{\text{tenure} + 1}$ | Captures long-term historical average monthly billing. |
| **`charge_diff_from_avg`** | $\text{MonthlyCharges} - \text{avg\_monthly\_spend}$ | Detects recent price increases or plan upgrades relative to baseline. |
| **`total_addons_count`** | $\sum \mathbb{I}(\text{Service Addon} == \text{'Yes'})$ | Quantifies product adoption and overall platform stickiness. |
| **`is_long_term_contract`** | $\mathbb{I}(\text{Contract} \in [\text{'One year'}, \text{'Two year'}])$ | Encodes contractual retention commitments. |
