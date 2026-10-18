# Logistic Regression and Classification — Machine Learning Practice

## 📌 About This Repository

This repository contains my hands-on practice with **Logistic Regression and Classification** using Python and Scikit-learn. It covers binary classification, multiclass classification, imbalanced datasets, hyperparameter tuning, and model evaluation techniques.

The notebooks use synthetically generated datasets to explore classification algorithms, experiment with model parameters, and understand how different evaluation metrics help assess machine learning models.

## 📂 Project Structure

```text
Logistic-Regression/
│
├── Binary_classification.ipynb
├── logistic_imbalance_practice.ipynb
├── Multiclass_classification.ipynb
├── ROC_AUC.ipynb
└── README.md
```

## 📚 Notebooks Overview

### 1. Binary Classification
**File:** `Binary_classification.ipynb`

Topics covered:
- Generating a synthetic binary classification dataset using `make_classification()`.
- Splitting data into training and testing sets.
- Training a Logistic Regression model.
- Evaluating the model using accuracy, precision, recall, and F1-score.
- Hyperparameter tuning using `GridSearchCV`.
- Randomized hyperparameter search using `RandomizedSearchCV`.
- Evaluating predictions using a classification report and confusion matrix.

### 2. Logistic Regression with Imbalanced Data
**File:** `logistic_imbalance_practice.ipynb`

Topics covered:
- Creating a highly imbalanced binary classification dataset.
- Examining class distribution using `Counter`.
- Visualising class distribution with Seaborn.
- Training Logistic Regression on imbalanced data.
- Exploring `class_weight` to account for unequal class frequencies.
- Tuning model parameters using `GridSearchCV` and `RandomizedSearchCV`.
- Evaluating model performance using accuracy, precision, recall, F1-score, and a confusion matrix.

### 3. Multiclass Classification
**File:** `Multiclass_classification.ipynb`

Topics covered:
- Generating a synthetic dataset with three classes.
- Implementing Logistic Regression using `OneVsRestClassifier`.
- Training and testing a multiclass classification model.
- Tuning estimator parameters using `GridSearchCV`.
- Understanding nested parameter names such as `estimator__C` and `estimator__solver`.
- Evaluating model performance using accuracy, classification reports, and confusion matrices.

### 4. ROC Curve and AUC
**File:** `ROC_AUC.ipynb`

Topics covered:
- Generating an imbalanced binary classification dataset.
- Training a Logistic Regression model.
- Obtaining class probabilities using `predict_proba()`.
- Calculating the ROC-AUC score using `roc_auc_score()`.
- Generating ROC curve coordinates using `roc_curve()`.
- Comparing model scores with a dummy baseline.
- Plotting the ROC curve and annotating threshold values.
- Exploring the relationship between thresholds, true positive rate, and false positive rate.

## 🛠️ Technologies and Libraries

- **Python** — Programming language
- **NumPy** — Numerical computing
- **Pandas** — Data manipulation
- **Matplotlib** — Data visualisation
- **Seaborn** — Statistical visualisation
- **Scikit-learn** — Machine learning models, dataset generation, model selection, and evaluation

## 🧠 Key Concepts Learned

Through these notebooks, I practised:

- Binary and multiclass classification
- Logistic Regression
- One-vs-Rest classification
- Imbalanced datasets and class weights
- Train-test splitting
- Cross-validation using `StratifiedKFold`
- Hyperparameter optimisation with Grid Search and Randomized Search
- Classification metrics: accuracy, precision, recall, and F1-score
- Confusion matrix and classification report
- ROC curve, AUC score, and decision thresholds
- Probability-based predictions using `predict_proba()`

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project directory

```bash
cd <your-repository-folder>
```

### 3. Install the required libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open any notebook and run the cells to explore the code and results.

## 🎯 Learning Objective

The objective of this repository is to strengthen my practical understanding of classification algorithms, model evaluation, cross-validation, and hyperparameter tuning as part of my journey in **Data Science and Machine Learning**.

## 🚀 Future Improvements

- Practise classification on real-world datasets.
- Compare Logistic Regression with other classification algorithms.
- Explore additional techniques for handling imbalanced datasets.
- Improve model evaluation using precision-recall curves and other relevant metrics.
- Organise experiments and document findings more systematically.

---

**Author:** Ashish Singh

**Field:** Data Science and Machine Learning

*This repository is part of my ongoing hands-on learning journey in Python, Data Science, and Machine Learning.*

