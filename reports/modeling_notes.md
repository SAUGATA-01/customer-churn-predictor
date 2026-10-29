- **Best model:** XGBoost and Random Forest tied for the highest test ROC-AUC of 0.853.

- **Interpretable vs complex:** Logistic Regression achieved 0.848 test ROC-AUC, only 0.005 lower than the best models.

- **Top churn drivers:** Contract type (especially month-to-month), customer tenure, and internet service type (especially fiber optic) were important predictors.

- **Recall:** XGBoost achieved the highest recall at 79.1%, while Random Forest had the highest precision at 53.2%.

- **Overfitting:** The CV and test ROC-AUC differences were below 0.05 for all models, showing no strong overfitting signal from this check.