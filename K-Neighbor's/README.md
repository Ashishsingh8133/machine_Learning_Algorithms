# K-Nearest Neighbours (KNN) — Machine Learning Practice

## 📌 About This Repository

This repository contains my hands-on practice with the **K-Nearest Neighbours (KNN)** algorithm using Python and Scikit-learn.

It covers both classification and regression, including model training, prediction, train-test splitting, and performance evaluation using appropriate metrics.

The notebooks use synthetically generated datasets to practise the implementation of KNN models.

## 📂 Project Structure

```text
K-Nearest-Neighbours/
│
├── KNN_Classifier.ipynb
├── KNN_Regressor.ipynb
└── README.md
```

## 📚 Notebooks Overview

### 1. KNN Classifier

**File:** `KNN_Classifier.ipynb`

Topics covered:
- Generating a synthetic binary classification dataset using `make_classification()`.
- Creating a dataset with 1,000 samples and three features.
- Splitting the dataset into training and testing sets.
- Implementing `KNeighborsClassifier` from Scikit-learn.
- Training the model using `n_neighbors=5`.
- Generating predictions on the test dataset.
- Evaluating the model using a confusion matrix.
- Measuring accuracy using `accuracy_score`.
- Analysing precision, recall, and F1-score using `classification_report`.
- Exploring the use of `GridSearchCV` to identify the best value of K from 1 to 10 as a practice task.

### 2. KNN Regressor

**File:** `KNN_Regressor.ipynb`

Topics covered:
- Generating a synthetic regression dataset using `make_regression()`.
- Creating a dataset with 1,000 samples, two features, and noise.
- Splitting the dataset into training and testing sets.
- Implementing `KNeighborsRegressor` from Scikit-learn.
- Training the model using `n_neighbors=6`.
- Predicting continuous target values on the test dataset.
- Evaluating regression performance using the R² score.
- Calculating Mean Absolute Error (MAE).
- Calculating Mean Squared Error (MSE).

## 🛠️ Technologies and Libraries

- **Python** — Programming language
- **NumPy** — Numerical computing
- **Pandas** — Data manipulation
- **Matplotlib** — Data visualisation
- **Seaborn** — Statistical visualisation
- **Scikit-learn** — Dataset generation, KNN models, train-test splitting, and evaluation metrics

## 🧠 Key Concepts Learned

Through these notebooks, I practised:

- K-Nearest Neighbours classification
- K-Nearest Neighbours regression
- Binary classification using synthetic data
- Regression using synthetic data
- Training and testing data splitting
- Choosing the number of neighbours (`n_neighbors`)
- Model fitting and prediction
- Classification evaluation metrics
- Regression evaluation metrics
- Confusion matrix and classification report
- Accuracy, precision, recall, and F1-score
- R² score, MAE, and MSE
- Hyperparameter optimisation as a next step using GridSearchCV

## 🎯 Learning Objective

The objective of this repository is to develop a practical understanding of KNN algorithms for classification and regression, learn how to evaluate model performance, and explore how the number of neighbours affects predictions.

## 🚀 Future Improvements

- Implement GridSearchCV to identify the best value of K for classification and regression.
- Compare model performance across different values of `n_neighbors`.
- Explore feature scaling and its effect on KNN performance.
- Experiment with different distance metrics and neighbour weighting.
- Apply KNN to real-world datasets.
- Compare KNN with other machine learning algorithms.

---

**Author:** Ashish Singh

**Field:** Data Science and Machine Learning

*This repository is part of my ongoing hands-on learning journey in Python, Data Science, and Machine Learning.*
