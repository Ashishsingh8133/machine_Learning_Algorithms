# Support Vector Machines (SVM) — Machine Learning Practice

## 📌 About This Repository

This repository contains my hands-on practice with **Support Vector Machines (SVM)** using Python and Scikit-learn. It covers Support Vector Classification, Support Vector Regression, different SVM kernels, feature engineering, model evaluation, and hyperparameter tuning.

The notebooks demonstrate how SVM models can be applied to classification and regression problems using synthetic data and the Seaborn Tips dataset.

## 📂 Project Structure

```text
Support-Vector-Machines/
│
├── Linear_Support_Vector_Classifier.ipynb
├── Support_Vector_Regressor.ipynb
├── SVM_Kernels.ipynb
└── README.md
```

## 📚 Notebooks Overview

### 1. Linear Support Vector Classifier

**File:** `Linear_Support_Vector_Classifier.ipynb`

Topics covered:
- Generating a synthetic binary classification dataset using `make_classification()`.
- Visualising data points using Seaborn scatter plots.
- Splitting data into training and testing sets.
- Training a Support Vector Classifier using `SVC`.
- Experimenting with different kernels: linear, RBF, polynomial, and sigmoid.
- Comparing classification performance using classification reports and confusion matrices.
- Hyperparameter tuning using `GridSearchCV`.
- Exploring the `C`, `gamma`, and `kernel` parameters.
- Evaluating the best model selected through cross-validation.

### 2. Support Vector Regression

**File:** `Support_Vector_Regressor.ipynb`

Topics covered:
- Loading and exploring the Seaborn Tips dataset.
- Selecting features to predict the `total_bill` target.
- Exploring categorical variables and their distributions.
- Splitting data into training and testing sets.
- Encoding categorical variables using `LabelEncoder` and `OneHotEncoder`.
- Applying `ColumnTransformer` for feature preprocessing.
- Training a Support Vector Regressor using `SVR`.
- Evaluating regression performance using the R² score.
- Hyperparameter tuning using `GridSearchCV` with five-fold cross-validation.
- Experimenting with the RBF kernel and different values of `C` and `gamma`.

### 3. SVM Kernels and Feature Engineering

**File:** `SVM_Kernels.ipynb`

Topics covered:
- Creating a synthetic dataset with concentric circular classes.
- Visualising data points using Matplotlib.
- Creating polynomial features from existing features.
- Exploring squared features and interaction terms.
- Understanding how feature transformations can help represent nonlinear relationships.
- Visualising transformed features using interactive 3D scatter plots with Plotly.
- Training an SVM classifier using the RBF kernel.
- Evaluating classification performance using a classification report and confusion matrix.

## 🛠️ Technologies and Libraries

- **Python** — Programming language
- **NumPy** — Numerical computing and array manipulation
- **Pandas** — Data manipulation and analysis
- **Matplotlib** — Data visualisation
- **Seaborn** — Statistical visualisation and sample datasets
- **Plotly Express** — Interactive 3D visualisation
- **Scikit-learn** — SVM models, preprocessing, dataset generation, model selection, and evaluation

## 🧠 Key Concepts Learned

Through these notebooks, I practised:

- Support Vector Classification (SVC)
- Support Vector Regression (SVR)
- Linear, polynomial, RBF, and sigmoid kernels
- Nonlinear classification
- Feature engineering and polynomial feature transformations
- Categorical feature encoding
- Training and testing data splitting
- Cross-validation using `GridSearchCV`
- Hyperparameter tuning with `C` and `gamma`
- Classification reports and confusion matrices
- Regression evaluation using the R² score

## 🎯 Learning Objective

The objective of this repository is to develop a practical understanding of SVM algorithms, kernel functions, feature transformations, regression, classification, and hyperparameter optimisation as part of my ongoing Data Science and Machine Learning journey.

## 🚀 Future Improvements

- Apply SVM models to real-world datasets.
- Compare SVM with Logistic Regression, Decision Trees, and other algorithms.
- Explore feature scaling and its effect on SVM performance.
- Experiment with different kernel parameters and regularisation strengths.
- Evaluate regression models using additional metrics such as MAE and RMSE.
- Document and compare the performance of different models systematically.

---

**Author:** Ashish Singh

**Field:** Data Science and Machine Learning

*This repository is part of my ongoing hands-on learning journey in Python, Data Science, and Machine Learning.*

