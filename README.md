# 📊 Customer Churn Forecasting Project

## 1. Executive Summary
This project delivers an end-to-end Machine Learning pipeline designed to forecast customer churn in a telecommunications dataset. By transforming raw account profiles through domain-specific feature engineering and benchmarking five distinct classification architectures (**Logistic Regression, Random Forest, XGBoost, Support Vector Machine, and MLP Neural Network**), the finalized model achieves a **Tuned ROC-AUC of 0.8462** and a **Churn Recall of 0.78 (78%)**. The full workflow is packaged into a production-ready deployment class (`ChurnPredictorPipeline`) using serialized `joblib` artifacts.

---

## 2. Dataset Overview & Preprocessing Pipeline

### 2.1 Dataset Profile
- **Total Records:** 7,043 customer accounts
- **Total Features:** 21 initial columns (18 categorical, 3 numerical)
- **Target Variable:** `Churn` (`Yes` / `No`, mapped to `1` / `0`)
- **Initial Class Distribution:** 73.5% Retained (`0`), 26.5% Churned (`1`)

### 2.2 Preprocessing & Data Cleaning
1. **Identifier Elimination:** Dropped non-predictive attributes (`customerID`, `gender`, `PhoneService`) to eliminate noise and prevent data leakage.
2. **Data Type Casting & Missing Value Imputation:** 
   - `TotalCharges` was loaded as a string due to blank space entries (`" "`).
   - Coerced blank entries to `NaN` using `pd.to_numeric(..., errors='coerce')` and imputed missing values using `MonthlyCharges` (corresponding to new accounts with `tenure = 0`).
3. **Outlier Mitigation & Normalization:** Applied `StandardScaler` to continuous variables (`tenure`, `MonthlyCharges`, `TotalCharges`) to standardize features to zero-mean unit variance for distance-based, linear, and neural network algorithms.
4. **Addressing Class Imbalance:** Rather than generating synthetic samples (e.g., SMOTE), cost-sensitive learning settings were used natively in estimator loss functions:
   - `class_weight='balanced'` in Logistic Regression, Random Forest, and Support Vector Machine.
   - `scale_pos_weight = ratio_negative_to_positive` in XGBoost.

---

## 3. Feature Engineering Strategy

To capture non-linear customer behaviors and financial friction, six domain-specific continuous and categorical metrics were engineered:

| Feature Name | Mathematical / Logical Formula | Business Rationale & Behavioral Impact |
| :--- | :--- | :--- |
| **`log_tenure`** | $\log(1 + \text{tenure})$ | Normalizes tenure skewness and models diminishing churn risk as account longevity grows. |
| **`tenure_cohort`** | `[onboarding, established, loyal, veteran]` | Groups accounts into distinct account lifecycle stages ($0\text{--}3$, $4\text{--}12$, $13\text{--}36$, and $36+$ months). |
| **`avg_monthly_spend`** | $\frac{\text{TotalCharges}}{\text{tenure} + 1}$ | Computes the baseline historical average monthly spend per account. |
| **`charge_diff_from_avg`** | $\text{MonthlyCharges} - \text{avg\_monthly\_spend}$ | Detects recent price hikes, plan upgrades, or bill shocks relative to baseline. |
| **`total_addons_count`** | $\sum \mathbb{I}(\text{Service Addon} == \text{'Yes'})$ | Quantifies overall account engagement and platform stickiness across protective services (`OnlineSecurity`, `TechSupport`, `DeviceProtection`). |
| **`is_long_term_contract`** | $\mathbb{I}(\text{Contract} \in [\text{'One year'}, \text{'Two year'}])$ | Encodes formal contractual commitment to differentiate flexible vs. locked-in accounts. |

---

## 4. Modeling Methodology & Selection

### 4.1 Benchmark Evaluation
Five algorithms were trained and benchmarked using identical **80/20 stratified train-test splits**:

| Model Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** 🏆 | **0.7445** | **0.5122** | **0.7861** | **0.6203** | **0.8476** |
| **Neural Network (MLP)** | 0.7949 | 0.6239 | 0.5722 | 0.5969 | 0.8365 |
| **Support Vector Machine (SVM)** | 0.7587 | 0.5306 | 0.7888 | 0.6344 | 0.8289 |
| **Random Forest** | 0.7878 | 0.6335 | 0.4759 | 0.5435 | 0.8259 |
| **XGBoost** | 0.7708 | 0.5586 | 0.6497 | 0.6007 | 0.8255 |

### 4.2 Top Performer Selection & Hyperparameter Optimization
- **Selected Model:** **Logistic Regression** achieved the top baseline performance on **ROC-AUC (0.8476)**.
- **GridSearch Optimization:** Optimized regularization parameters over 5 cross-validation folds:
  - Best Hyperparameters: `{'C': 10.0, 'solver': 'lbfgs'}`
  - Final Tuned ROC-AUC: **`0.8462`**

---

## 5. Evaluation Metrics Justification

1. **Recall / Sensitivity (Primary Metric):** In subscription churn, **False Negatives** (failing to identify an actual churner) result in total loss of Customer Lifetime Value (CLV). **False Positives** (sending a retention offer to a loyal user) incur only minor promotional costs. Setting an optimal probability decision threshold at **$0.40$** captures **78% of all true churners**.
2. **ROC-AUC:** Evaluates probability calibration and ranking capabilities across all classification thresholds without bias toward majority class prevalence.
3. **Precision & F1-Score:** Monitored to maintain marketing campaign efficiency and prevent excessive false alarm rates.

---

## 6. Model Persistence & Production Deployment
The full pipeline is packaged into `ChurnPredictorPipeline` in `src/inference_pipeline.py`. Model state and preprocessors are saved under `artifacts/`:
- `churn_model.pkl`: Hyperparameter-tuned Logistic Regression model.
- `scaler.pkl`: Fitted `StandardScaler` instance.
- `feature_names.pkl`: Ordered feature column array ensuring seamless input validation.
