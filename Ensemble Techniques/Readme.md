# Machine Learning Projects: Classification & Regression

## 📌 Overview

This repository contains two end-to-end Machine Learning projects demonstrating the implementation of supervised learning algorithms for both classification and regression problems.

The projects cover essential Machine Learning workflows, including data preprocessing, exploratory data analysis, feature engineering, model selection, model evaluation, and hyperparameter tuning using Scikit-learn.

## 📂 Projects Included

### 1. Classification – Travel Package Prediction

**Objective:** Build machine learning classification models to predict whether a customer will purchase a travel package (`ProdTaken`).

**Key Steps:**
- Data loading and initial exploration using Pandas.
- Missing value identification and imputation.
- Feature engineering by creating a `TotalVisiting` feature.
- Identification of numerical, categorical, discrete, and continuous features.
- Removal of unnecessary features such as `CustomerID`.
- Train-test splitting.
- Feature scaling using StandardScaler.
- Categorical feature encoding using OneHotEncoder.
- Model training and comparison.
- Hyperparameter tuning using RandomizedSearchCV.
- ROC curve and AUC evaluation.

**Algorithms Implemented:**
- AdaBoost Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- Random Forest Classifier

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC Score

### 2. Regression – Car Price Prediction

**Objective:** Develop machine learning regression models to predict car prices using different vehicle characteristics.

**Key Steps:**
- Data loading and initial exploration.
- Data cleaning and removal of unnecessary features.
- Identification of numerical and categorical features.
- Feature encoding using LabelEncoder and OneHotEncoder.
- Feature scaling using StandardScaler.
- Train-test splitting.
- Training and comparison of multiple regression algorithms.
- Hyperparameter tuning using RandomizedSearchCV.
- Model performance evaluation.

**Algorithms Implemented:**
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet Regression
- Decision Tree Regressor
- Random Forest Regressor
- K-Nearest Neighbors Regressor

**Evaluation Metrics:**
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## 🛠️ Technologies & Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn
- XGBoost
- Jupyter Notebook

## 🎯 Learning Outcomes

Through these projects, I gained practical experience in:

- Understanding and implementing supervised machine learning algorithms.
- Handling missing values and performing feature engineering.
- Applying feature scaling and encoding techniques.
- Comparing multiple machine learning algorithms.
- Evaluating classification and regression models.
- Optimising model performance using hyperparameter tuning.
- Understanding model performance through appropriate evaluation metrics.

---

**Author:** Ashish Kumar Singh

*Continuously learning and building practical Machine Learning projects.*
