# E-commerce Customer Churn Risk Prediction

**🔍 Introduction**  

In today’s highly competitive e-commerce landscape, especially in India, customer churn has become a serious challenge. Many customers make only one purchase and never return, making it harder for businesses to build long-term relationships. This project aims to help e-commerce businesses minimize churn and improve customer loyalty through predictive analytics.

**📌 Problem Statement**  

Based on the dataset used in this project, the churn rate is very high (58.37%). This project defines churn as no transactions within the last six months, making it easier to identify truly inactive customers. Early detection of high-risk customers is essential to maintain profitability.

**🎯 Goals & Objectives**

- Predict customer churn risk to support personalized marketing strategies.
- Segment customers based on their churn risk scores.
- Uncover behavioral factors that drive churn.
- Evaluate model performance with Recall ≥ 70% & ROC-AUC ≥ 70%.
- Provide monthly updates to support ongoing marketing strategy improvements.

**🔗 ERD & Data Pre-Processing**  

This project integrates various e-commerce tables — orders, order items, customers, products, and sellers — connected through primary and foreign keys. The data pipeline handles missing values, data type conversion, and feature engineering to improve data quality and model performance.

**📊 EDA & Feature Engineering**  

Comprehensive univariate and multivariate analyses are used to identify churn patterns based on region, spending category, payment type, product category, and review scores.

**⚙️ Modeling**  

- Model: XGBoost
- Handling imbalance: SMOTE
- Evaluation metrics: Recall & ROC-AUC
- Hyperparameter tuning: GridSearchCV
- Performance: Recall 88%, ROC-AUC 71%

**Justification for Model Selection**
- **Why XGBoost?**
  - Best recall with consistent test results
  - Well-suited for imbalanced classification
  - Supports numerical and categorical data
  - Robust tuning via GridSearchCV
 
**📈 Business Impact**

- Potential churn reduction of 16.43%
- Projected revenue retention of $4.92 million with low-cost interventions
- Significant ROI through proactive churn prediction and retention strategies

**📌 Deployment**

- Streamlit App: https://ashirvada-churn-risk-prediction.streamlit.app/
- Looker Studio Dashboard: https://lookerstudio.google.com/reporting/76834095-8e84-4141-9bed-7ea2f1757297/page/0VZKF

**🛠️ Tools & Tech Stack**

- Python (Jupyter Notebook)
- Streamlit for app deployment
- Google Looker Studio for dashboard & reporting

**📚 References**

- Saha et al. (2023). Deep churn prediction method for telecommunication industry.
- Thomas et al. (2004). Recapturing Lost Customers.
- Little & Rubin (2002). Statistical Analysis with Missing Data.


---

