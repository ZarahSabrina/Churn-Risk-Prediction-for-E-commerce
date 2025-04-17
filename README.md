# Group4-churn-prediction

#  Customer Churn Prediction

## 1. Project Overview
- **Goal:** Predict customer churn in an e-commerce setting.
- **Type of Problem:** Binary classification (`Churn = 1`, `Not Churn = 0`)
- **Business Impact:** Helps retain customers by identifying those at risk of churn.

---

## 2.  Data Preprocessing
- **Missing Value Handling:** Dropped or filled with appropriate values.
- **Feature Selection:** Removed irrelevant columns like IDs and timestamps.
- **Feature Engineering:**
  - Created `spending_category` from `payment_value`
  - One-hot encoding for `customer_region`
- **Outlier Handling:** Outliers retained to preserve behavioral signals.
- **Log Transformation:** Applied to `price` and `payment_value` to reduce skewness.
- **Scaling:** StandardScaler used after log transformation.

---

## 3. Handling Class Imbalance
- **Issue:** Target variable imbalance (~58% churn vs 42% not churn)
- **Solution:** Used SMOTE to balance the classes in the training set.

---

## 4. Model Pipeline
- **Train-Test Split:** 80:20 with `stratify=y`
- **Baseline Model:** Logistic Regression
- **Models Evaluated:**
  - Random Forest
  - XGBoost
  - Decision Tree
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

| Model                 | Recall (Test) | ROC AUC (Test) |
|----------------------|---------------|----------------|
| Logistic Regression  | 0.4526        | 0.5255         |
| Random Forest        | 0.7800        | 0.8283         |
| Tuned XGBoost        | 0.8788        | 0.7417         |
| Ensemble             | 0.9813        | 0.6753         |


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
 
