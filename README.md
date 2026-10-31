# Telecom Customer Churn Prediction: Technical Project Report

## 1. Executive Summary
This project delivers an end-to-end Machine Learning pipeline to predict customer churn in a telecommunications dataset. By processing raw customer profiles through domain-specific feature engineering and benchmarking seven distinct classification architectures, the finalized model achieves a **Tuned ROC-AUC of 0.8462** and a **Churn Recall of 0.78 (78%)**. The pipeline is packaged into a production-ready deployment class (`ChurnPredictorPipeline`) using serialized `joblib` artifacts.

---

## 2. Dataset Overview & Preprocessing Pipeline

### 2.1 Dataset Profile
- **Total Records:** 7,043 customer accounts
- **Total Features:** 21 initial columns (18 categorical, 3 numerical)
- **Target Variable:** `Churn` (`Yes` / `No`, converted to `1` / `0`)
- **Initial Class Distribution:** 73.5% Retained (`0`), 26.5% Churned (`1`)

### 2.2 Preprocessing & Data Cleaning
1. **Identifier Removal:** Dropped non-predictive keys (`customerID`, `gender`, `PhoneService`) to eliminate noise and potential leakage.
2. **Data Type Casting & Imputation:** 
   - `TotalCharges` was imported as string/object due to blank space strings (`" "`).
   - Coerced blank entries to `NaN` via `pd.to_numeric()` and imputed missing values using `MonthlyCharges` (corresponding to new customers with zero tenure).
3. **Outlier Mitigation & Feature Normalization:** Applied `StandardScaler` to continuous variables (`tenure`, `MonthlyCharges`, `TotalCharges`) to standardize zero-mean unit variance for distance and linear models.
4. **Addressing Class Imbalance:** Rather than generating synthetic data (e.g., SMOTE), class imbalance was handled natively using cost-sensitive learning settings:
   - `class_weight='balanced'` in Logistic Regression, Random Forest, and SVM.
   - `scale_pos_weight = ratio_negative_to_positive` in XGBoost and LightGBM.

---

## 3. Feature Engineering Strategy

To capture non-linear customer behaviors, several domain-specific metrics were created:

| Feature Name | Mathematical / Logical Formula | Business Rationale |
| :--- | :--- | :--- |
| **`log_tenure`** | $\log(1 + \text{tenure})$ | Reduces skewness and captures diminishing returns of customer longevity. |
| **`tenure_cohort`** | Binned into `[onboarding, established, loyal, veteran]` | Groups customers into distinct account lifecycle stages. |
| **`avg_monthly_spend`** | $\frac{\text{TotalCharges}}{\text{tenure} + 1}$ | Measures historical average billing per active month. |
| **`charge_diff_from_avg`** | $\text{MonthlyCharges} - \text{avg\_monthly\_spend}$ | Detects recent price spikes relative to historical averages. |
| **`total_addons_count`** | $\sum \mathbb{I}(\text{Service Addon} == \text{'Yes'})$ | Quantifies overall account engagement and product stickiness. |
| **`is_long_term_contract`** | $\mathbb{I}(\text{Contract} \in [\text{'One year'}, \text{'Two year'}])$ | Binary flag for contractual retention commitment. |

---

## 4. Modeling Methodology & Selection

### 4.1 Benchmark Evaluation
Seven algorithms were evaluated under identical 80/20 stratified train-test splits:

| Model Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.7445** | **0.5122** | **0.7861** | **0.6203** | **0.8476** |
| **CatBoost** | 0.7672 | 0.5444 | 0.7540 | 0.6323 | 0.8380 |
| **Neural Network (MLP)** | 0.7949 | 0.6239 | 0.5722 | 0.5969 | 0.8365 |
| **LightGBM** | 0.7509 | 0.5216 | 0.7433 | 0.6130 | 0.8316 |
| **Support Vector Machine** | 0.7587 | 0.5306 | 0.7888 | 0.6344 | 0.8289 |
| **Random Forest** | 0.7878 | 0.6335 | 0.4759 | 0.5435 | 0.8259 |
| **XGBoost** | 0.7708 | 0.5586 | 0.6497 | 0.6007 | 0.8255 |

### 4.2 Top Performer Selection & Hyperparameter Tuning
- **Selected Model:** **Logistic Regression** achieved the highest baseline ROC-AUC (0.8476).
- **GridSearch Optimization:** Optimized over regularized cost parameters:
  - Best Parameters: `{'C': 10.0, 'solver': 'lbfgs'}`
  - Final Tuned ROC-AUC: **0.8462**

---

## 5. Evaluation Metrics Justification

1. **Recall (Primary Metric):** In customer churn, False Negatives (missing an actual churner) are vastly more expensive than False Positives (sending a retention offer to a loyal customer). The final model captures **78% of all true churners**.
2. **ROC-AUC:** Evaluates probability calibration and class separation across all decision thresholds.
3. **F1-Score:** Balances Precision and Recall to ensure campaign spending efficiency.

---

## 6. Model Persistence & Production Deployment
The full pipeline is packaged into `ChurnPredictorPipeline` in `src/inference_pipeline.py`. Model state and preprocessors are saved under `artifacts/`:
- `churn_model.pkl`: Hyperparameter-tuned Logistic Regression model.
- `scaler.pkl`: Fitted `StandardScaler` instance.
- `feature_names.pkl`: Ordered feature column list ensuring seamless input validation.
