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
-Performed data cleaning and exploratory data analysis (EDA).
-Identified numerical, categorical, discrete and continuous features.
-Removed unnecessary features such as car_name and brand.
-Applied Label Encoding to the model feature.
-Implemented ColumnTransformer for preprocessing numerical and categorical features.
-Used StandardScaler for numerical feature scaling and OneHotEncoder for categorical feature encoding.
-Split the dataset into training and testing sets (80:20).
-Trained and compared multiple regression algorithms.
-Implemented RandomizedSearchCV with 5-fold cross-validation for hyperparameter tuning.
-Evaluated model performance using R² Score, MAE, MSE and RMSE.

**Algorithms Implemented:**
-Decision Tree Regressor
-Random Forest Regressor
-Gradient Boosting Regressor
-K-Nearest Neighbours Regressor
-XGBoost Regressor

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
