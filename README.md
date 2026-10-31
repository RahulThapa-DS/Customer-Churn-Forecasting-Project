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

---

## 📈 Model Benchmarking & Tuning

Seven machine learning models were trained and evaluated using identical **80/20 stratified train-test splits**:

| Model Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** 🏆 | **0.7445** | **0.5122** | **0.7861** | **0.6203** | **0.8476** |
| **CatBoost** | 0.7672 | 0.5444 | 0.7540 | 0.6323 | 0.8380 |
| **Neural Network (MLP)** | 0.7949 | 0.6239 | 0.5722 | 0.5969 | 0.8365 |
| **LightGBM** | 0.7509 | 0.5216 | 0.7433 | 0.6130 | 0.8316 |
| **Support Vector Machine (SVM)** | 0.7587 | 0.5306 | 0.7888 | 0.6344 | 0.8289 |
| **Random Forest** | 0.7878 | 0.6335 | 0.4759 | 0.5435 | 0.8259 |
| **XGBoost** | 0.7708 | 0.5586 | 0.6497 | 0.6007 | 0.8255 |

### Selected Model & Hyperparameter Optimization
**Logistic Regression** achieved the highest baseline ROC-AUC (0.8476). Applying 5-fold cross-validated `GridSearchCV` optimized regularization parameters:
- **Best Parameters:** `{'C': 10.0, 'solver': 'lbfgs'}`
- **Final Tuned ROC-AUC:** **`0.8462`**

---

## 🎯 Metrics Justification & Business Insights

### Why Prioritize Recall?
In churn prediction, **False Negatives** (failing to identify a churning customer) result in total loss of Customer Lifetime Value (CLV). **False Positives** (sending a discount offer to a loyal customer) carry only minor promotional costs. Thus, the model prioritizes **Recall (78%)** by setting an optimal decision threshold at **$0.40$**.

### Top Churn Risk Drivers
- **Month-to-Month Contracts:** Strongest indicator of short-term cancellation risk.
- **Fiber Optic Service without Add-ons:** High monthly costs coupled with lack of security/tech support drives churn.
- **Recent Bill Price Surges (`charge_diff_from_avg`):** Price jumps relative to historical averages trigger billing friction.
- **Electronic Check Payments:** Manual payment methods lack automatic renewals and encourage monthly re-evaluation.

### Recommended Retention Actions
- **Contract Migration:** Offer small bill credits ($5–$10) to month-to-month users who transition to 1-year plans.
- **Onboarding Assistance:** Implement automated check-ins during the critical 0–3 month onboarding window (`onboarding` cohort).
- **Auto-Pay Incentives:** Provide a one-time account credit for users adopting automated credit card/bank payments.

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
# Clone repository
git clone [https://github.com/YourUsername/telecom-churn-prediction.git](https://github.com/YourUsername/telecom-churn-prediction.git)
cd telecom-churn-prediction

# Install requirements
pip install -r requirements.txt
```
### 2. Running Inference in Python
```
import pandas as pd
from src.inference_pipeline import ChurnPredictorPipeline

# Load production wrapper
pipeline = ChurnPredictorPipeline(artifacts_dir="artifacts")

# Pass raw customer dictionary / DataFrame
raw_input = pd.DataFrame([{
    "customerID": "1234-XYZ",
    "tenure": 2,
    "MonthlyCharges": 85.50,
    "TotalCharges": "171.00",
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check"
}])

# Get predictions
results = pipeline.predict(raw_input, threshold=0.40)
print(results)
# Output:
#  churn_probability  churn_prediction risk_level
#             0.8250                 1  High Risk
```
