# 🏦 Loan Default Prediction

> **Predicting which borrowers are at risk of defaulting on their loans using machine learning.**  
> Built with Python, scikit-learn
---

## 📌 Project Overview

Financial institutions lose billions annually to loan defaults. This project builds a binary classification system to identify high-risk borrowers **before** a loan is approved — enabling smarter lending decisions, better risk-based pricing, and reduced financial exposure.

The project walks through the full data science workflow: data cleaning, exploratory analysis, feature engineering, model building, leakage detection, and evaluation — demonstrating the kind of end-to-end thinking that translates directly to real-world credit risk roles.

---

## 📂 Dataset

- **Source:** [Kaggle — Loan Default Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default)
- **Size:** ~148,000 rows × 34 columns
- **Target Variable:** `status` (0 = No Default, 1 = Default)
- **Features include:** loan amount, income, credit score, LTV ratio, debt-to-income ratio, loan type, region, property value, and more

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| pandas & numpy | Data manipulation |
| matplotlib & seaborn | Visualizations |
| scikit-learn | Preprocessing, modeling, evaluation |
| XGBoost | Gradient boosting model |
| Jupyter Notebook | Development environment |

---

## 📁 Project Structure

```
loan-default-prediction/
│
├── data/
│   └── Loan_Default.csv          # Raw dataset (download from Kaggle)
│
├── Loan_Default_Prediction.ipynb # Main notebook (full workflow)
└── README.md
```

---

## 🔍 Project Workflow

### 1. Data Cleaning
- Dropped non-informative ID column
- Standardized column names (lowercase, underscores)
- Cleaned inconsistent categorical values (`Sex Not Available` → `Unknown`)
- Created **missing value indicator flags** for columns with >15% missingness before imputing — capturing the signal that missingness itself carries
- Imputed numerical columns with median, categorical with mode

### 2. Exploratory Data Analysis
Key findings from EDA:

- **Class imbalance:** ~75% non-default vs ~25% default — informed decision to use `class_weight='balanced'` and prioritize ROC-AUC over accuracy
- **Loan amount & income** show similar distributions across both classes — raw financial size is a weak standalone predictor
- **Credit score** is uniformly distributed (500–900) across both classes — confirms the model cannot rely on it alone
- **EQUI credit type** showed a near-100% default rate — identified as a data anomaly and excluded from modeling
- **Default rate by category** (not raw counts) revealed that loan type2, purpose p2, and North-East region carry disproportionately higher risk despite lower volume
- **Correlation heatmap** revealed that missing value flags were perfectly correlated with the target — a data leakage issue that was identified and resolved

### 3. Feature Engineering
Three ratio-based features were engineered to capture financial risk more precisely than raw values:

```python
df['loan_to_income']  = df['loan_amount'] / (df['income'] + 1)
df['income_to_debt']  = df['income'] / (df['dtir1'] + 1)
df['credit_to_ltv']   = df['credit_score'] / (df['ltv'] + 1)
```

- LTV outliers capped at the 99th percentile
- Log-transformation applied to `loan_amount`, `income`, and `loan_to_income` to normalize skewed distributions
- Both `loan_to_income` and `log_loan_to_income` ranked in the top 15 features for both tree models — validating the engineering

### 4. Encoding & Preprocessing
- Used **sklearn's TargetEncoder** — replaces each category with its mean default rate, directly encoding risk into the feature rather than using arbitrary ordinal values
- Critically, encoding was **fit on training data only** and applied to the test set — preventing data leakage
- All numerical features standardized with `StandardScaler`

### 5. Data Leakage Detection & Resolution
> ⚠️ **This is the most important section of the project.**

Initial models returned a perfect AUC = 1.0 — a clear red flag. Correlation analysis identified two sources of leakage:

1. **Missing indicator flags** (`interest_rate_spread_missing`, `rate_of_interest_missing`, `upfront_charges_missing`) were perfectly correlated with the target — the pattern of missing financial data directly identified defaulters
2. **EQUI credit type** had a near-100% default rate — a data anomaly, not a generalizable signal
3. **Source columns** (`interest_rate_spread`, `rate_of_interest`, `upfront_charges`) were themselves leaking through their values

All leaking columns were dropped. Post-fix correlations confirmed no feature exceeded 0.20 correlation with the target.

**Identifying and resolving data leakage is a critical real-world skill** that many practitioners miss entirely.

### 6. Modeling
Three models were trained with class imbalance handling:

| Model | Class Imbalance Strategy |
|---|---|
| Logistic Regression | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced'` |
| XGBoost | `scale_pos_weight` (ratio of majority to minority class) |

---

## 📊 Results

### Model Performance

| Model | Accuracy | Default Precision | Default Recall | Default F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.71 | 0.44 | 0.66 | 0.53 | 0.7575 |
| Random Forest | 0.89 | 0.95 | 0.60 | 0.73 | 0.8879 |
| **XGBoost** | **0.88** | **0.76** | **0.73** | **0.75** | **0.8952** |

### ROC Curve
XGBoost achieved the best AUC at **0.8952**, outperforming both Logistic Regression (0.76) and Random Forest (0.89) on balanced default detection.

### Confusion Matrix Highlights
| Model | False Negatives (Missed Defaults) | False Positives |
|---|---|---|
| Logistic Regression | 2,524 | 6,135 |
| Random Forest | 2,956 | 252 |
| XGBoost | 1,958 | 1,677 |

**XGBoost selected as the final model** — it achieves the best balance between catching defaulters (recall) and minimizing false alarms (precision). In credit risk, missing a defaulter is costlier than over-flagging a good borrower, making recall on the default class the priority metric.

### Top Predictive Features (XGBoost & Random Forest agree)
1. **LTV (Loan-to-Value ratio)** — borrowers with low equity carry highest risk
2. **property_value** — collateral quality matters
3. **dtir1 (debt-to-income ratio)** — confirms debt burden outweighs raw income as a risk signal
4. **loan_to_income** *(engineered)* — validated the feature engineering effort
5. **lump_sum_payment & neg_ammortization** — loan structure carries as much signal as borrower demographics

---

## 💼 Key Business Insights & Recommended Solutions

### 1. 🚨 High Overall Default Rate (24.6%)
**Problem:** 1 in 4 borrowers defaulted — far above the industry standard of 2–10%, indicating the portfolio is heavily exposed to high-risk borrowers.

**Solution:** Implement a **pre-screening risk score** at the application stage using the XGBoost model (AUC = 0.90) to flag high-risk applicants before approval, directly reducing portfolio default exposure.

---

### 2. 💸 Debt Burden Matters More Than Income
**Problem:** Raw income alone does not separate defaulters from non-defaulters — borrowers across all income levels defaulted at similar rates.

**Solution:** Shift underwriting focus from *how much a borrower earns* to *how much of their income is already committed to debt* (`dtir1`). Set stricter debt-to-income ratio thresholds as a hard approval gate.

---

### 3. 🏠 Low Equity Borrowers Are Highest Risk
**Problem:** LTV (Loan-to-Value ratio) was the top predictive feature — borrowers with little equity relative to their property value defaulted most frequently.

**Solution:** Require a **minimum equity threshold** (e.g., reject or flag loans where LTV exceeds 80–85%), or apply higher interest rates to high-LTV loans to compensate for the increased lender exposure.

---

### 4. ⚠️ Loan Type 2 Is a High-Risk Product
**Problem:** Type2 loans had a 35% default rate — the highest of all loan types — a risk that was invisible when looking at raw counts alone and only surfaced through default rate analysis.

**Solution:** Conduct a product-level audit of Type2 loans. Consider tightening approval criteria, increasing collateral requirements, or repricing the product to reflect its true risk profile.

---

### 5. 🔎 Loan Purpose P2 Is Low Volume but High Risk
**Problem:** Purpose p2 carried a 33% default rate despite being the smallest borrower segment — a classic case where aggregate numbers hide a dangerous pocket of risk.

**Solution:** Flag p2 purpose loans for **enhanced due diligence** at the application stage. A small high-default segment can still generate significant losses if left unmonitored.

---

### 6. 🗺️ North-East Region Has Elevated Default Risk
**Problem:** The North-East had the highest default rate (~30%) despite being the smallest region by loan volume, suggesting regional economic conditions are influencing repayment ability.

**Solution:** Implement **region-specific risk adjustments** in pricing models. Loans originated in the North-East should carry higher risk premiums or stricter approval criteria until the underlying drivers are understood.

---

### 7. 📋 Incomplete Documentation Predicts Default
**Problem:** Borrowers with missing values for interest rate, rate spread, and upfront charges were overwhelmingly more likely to default — missingness was a stronger predictor than most actual financial features.

**Solution:** Make **complete financial documentation a hard requirement** for loan approval. An applicant who cannot or will not provide financial information is itself a meaningful credit risk signal.

---

### 8. 📉 Credit Score Alone Is Unreliable
**Problem:** Median credit scores were nearly identical (~700) for both defaulters and non-defaulters, making it a weak standalone gatekeeper in this portfolio.

**Solution:** Replace single-metric credit score thresholds with a **composite risk score** combining LTV, dtir1, loan-to-income ratio, and loan type. These combinations predict default far more reliably than credit score alone.

---

### 9. 📝 Loan Structure Is As Important As Borrower Profile
**Problem:** Features like `lump_sum_payment`, `neg_ammortization`, and `interest_only` ranked highly in XGBoost — meaning *how* a loan is structured carries as much risk signal as *who* the borrower is.

**Solution:** Scrutinize loan structure terms during underwriting. Loans with negative amortization or lump sum payment structures should face additional risk review regardless of the borrower's credit profile.

---

### 🏆 Overall Recommendation
Deploy the **XGBoost model** (AUC = 0.90, Default Recall = 73%) as a **first-pass risk screening tool** integrated into the loan application pipeline. Lower the classification threshold from 0.5 to ~0.3 to prioritize catching defaulters over minimizing false alarms — in lending, a missed default is always more costly than a declined good application.

---

## 💡 Key Takeaways

- Raw financial figures (income, loan amount, credit score) are surprisingly weak standalone predictors — **ratio-based features outperform them**
- **Missingness is informative** — borrowers with incomplete financial documentation are more likely to default
- **Data leakage can silently destroy model validity** — always correlate features against the target before trusting perfect scores
- Evaluating models on **ROC-AUC and F1** rather than accuracy is essential for imbalanced classification problems

---


## 🔮 Future Work

- **Threshold tuning** — lowering XGBoost's classification threshold from 0.5 to ~0.3 to improve recall on the default class at the cost of some precision
- **SHAP values** — for individual prediction explainability, which is required in regulated lending environments
- **Cross-validation** — replace single train/test split with k-fold CV for more robust evaluation
- **Hyperparameter tuning** — GridSearchCV or Optuna for XGBoost optimization

---

> **Note:** The business solutions and recommendations provided in this project are based on 
> my own interpretation of the data and the knowledge I have gathered throughout this analysis. 
> They are not affiliated with any financial institution.



