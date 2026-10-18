# Decision Trees — Classification and Regression

## 📌 About This Repository

This repository contains my hands-on practice with **Decision Tree algorithms** using Python and Scikit-learn. It covers both classification and regression, including data preprocessing, model training, tree visualisation, model evaluation, and hyperparameter tuning.

The notebooks demonstrate how Decision Trees can be applied to classification and regression tasks using the Iris dataset and a diabetes prediction dataset.

## 📂 Project Structure

```text
Decision-Trees/
│
├── DT_classifier.ipynb
├── DT_Regressor.ipynb
├── iris.csv
├── diabetes_prediction_dataset.csv
└── README.md
```

## 📚 Notebooks Overview

### 1. Decision Tree Classifier

**File:** `DT_classifier.ipynb`

**Dataset:** Iris dataset

Topics covered:
- Loading and exploring the Iris dataset using Pandas.
- Checking for missing values and exploring unique target classes.
- Encoding flower species using `LabelEncoder`.
- Separating independent and dependent variables.
- Splitting the dataset into training and testing sets.
- Training a `DecisionTreeClassifier`.
- Visualising the decision tree using Scikit-learn.
- Understanding overfitting caused by a fully grown decision tree.
- Evaluating predictions using a confusion matrix and classification report.
- Applying pre-pruning techniques through hyperparameter constraints.
- Tuning model parameters using `GridSearchCV` with five-fold cross-validation.
- Experimenting with `criterion`, `splitter`, `max_depth`, and `max_features`.
- Evaluating the tuned model using a confusion matrix and classification report.

### 2. Decision Tree Regressor

**File:** `DT_Regressor.ipynb`

**Dataset:** `diabetes_prediction_dataset.csv`

Topics covered:
- Loading and exploring the diabetes prediction dataset.
- Inspecting dataset information, dimensions, unique values, and missing values.
- Encoding categorical features using `LabelEncoder`.
- Standardising numerical features using `StandardScaler`.
- Separating independent and dependent variables.
- Splitting the dataset into training and testing sets.
- Analysing feature correlations using a Seaborn heatmap.
- Training a `DecisionTreeRegressor`.
- Generating predictions on the test dataset.
- Evaluating the initial model using the R² score.
- Applying hyperparameter tuning using `GridSearchCV` with five-fold cross-validation.
- Experimenting with `criterion`, `splitter`, `max_depth`, and `max_features`.
- Evaluating the tuned model using R², Mean Squared Error (MSE), and Mean Absolute Error (MAE).
- Visualising the decision tree.

## 🛠️ Technologies and Libraries

- **Python** — Programming language
- **NumPy** — Numerical computing
- **Pandas** — Data manipulation and analysis
- **Matplotlib** — Data visualisation
- **Seaborn** — Statistical visualisation and correlation heatmaps
- **Scikit-learn** — Decision Tree models, preprocessing, model selection, and evaluation

## 🧠 Key Concepts Learned

Through these notebooks, I practised:

- Decision Tree Classification
- Decision Tree Regression
- Data preprocessing and categorical encoding
- Feature scaling using StandardScaler
- Training and testing data splitting
- Decision tree visualisation
- Understanding overfitting
- Pre-pruning using tree hyperparameters
- Hyperparameter tuning with GridSearchCV
- Five-fold cross-validation
- Classification evaluation using confusion matrices and classification reports
- Regression evaluation using R², MAE, and MSE

## 🎯 Learning Objective

The objective of this repository is to develop a practical understanding of Decision Tree algorithms, explore their applications in classification and regression, understand overfitting, and learn how hyperparameter tuning can influence model performance.

## 🚀 Future Improvements

- Compare Decision Trees with Random Forest and other machine learning algorithms.
- Experiment with different tree depths and pruning strategies.
- Evaluate classification performance using accuracy, precision, recall, and F1-score.
- Compare regression performance using R², MAE, and RMSE.
- Explore feature importance and its role in model interpretation.
- Validate the choice of model against the target variable's data type.
- Apply Decision Tree algorithms to additional real-world datasets.

---

**Author:** Ashish Singh

**Field:** Data Science and Machine Learning

*This repository is part of my ongoing hands-on learning journey in Python, Data Science, and Machine Learning.*

