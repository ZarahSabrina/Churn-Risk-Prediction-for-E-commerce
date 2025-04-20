# Group4-churn-prediction

#  Customer Churn Prediction

## 1. Project Overview
- **Goal:** Give Churn Risk Score Customer in an e-commerce.
- **Type of Problem:** Binary classification (`Churn = 1`, `Not Churn = 0`)
- **Business Impact:** Helps retain customers by identifying those at risk of churn.

---

## 2.  Data Preprocessing
- **Missing Value Handling:** Dropped or filled with appropriate values.
- **Feature Engineering:**
  - One-hot encoding for `customer_region`
- **Outlier Handling:** Outliers retained to preserve behavioral signals.
- **Log Transformation:** Applied to `mean_price` and `total_payment_value` to reduce skewness.

---

## 3. Handling Class Imbalance
- **Issue:** Target variable imbalance (~58% churn vs 42% not churn)
- **Solution:** Used SMOTE to balance the classes in the training set.

---

## 4. Model Pipeline
- **Train-Test Split:** 70:30 with `stratify=y`
- **Baseline Model:** Logistic Regression
- **Models Evaluated:**
  - Random Forest
  - XGBoost
  - AdaBoost
  - CatBoost
  - Ensemble Voting (Random Forest + XGBoost)

---

## 5. Hyperparameter Tuning
- **Approach:**
  - GridSearchCV (for XGBoost)
  - RandomizedSearchCV (for Random Forest)
- **Scoring Metric:** Recall
- **Threshold Adjustment:** Custom threshold set to 0.4 to maximize recall

---

## 6. Model Evaluation

Metrik Recall dan ROC AUC


✅ **Final Model:** Tuned XGBoost → Best balance between recall and generalization.

---

## 7. Version Control
- Project managed using Git
- Structure:
```
📦 churn-prediction-project
 ┣ 📁 data
 ┣ 📁 notebooks
 ┣ 📁 app
 ┣ 📄 README.md
 ┣ 📄 requirements.txt
```

---

## 8. Justification for Model Selection
- **Why XGBoost?**
  - Best recall with consistent test results
  - Well-suited for imbalanced classification
  - Supports numerical and categorical data
  - Robust tuning via GridSearchCV
 
