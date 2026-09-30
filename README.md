# Linear Regression in Python

A practical machine learning project covering **Simple Linear Regression, Multiple Linear Regression, and Polynomial Regression** using Python and Scikit-learn.

This repository demonstrates the complete regression workflow, starting from understanding datasets and visualising relationships to training models, making predictions, evaluating performance, and exploring polynomial features and pipelines.

---

## 📌 Project Overview

Regression is one of the fundamental concepts in Machine Learning and is widely used for predicting continuous numerical values.

In this project, I explored different regression techniques and compared how they perform on different types of relationships between independent and dependent variables.

The project covers:

* Simple Linear Regression
* Multiple Linear Regression
* Polynomial Regression
* Polynomial Features
* Polynomial Regression using Pipeline
* Train/Test Split
* Cross-Validation
* Model Coefficients and Intercept
* Residual Analysis
* R² Score
* Adjusted R² Score
* MAE
* MSE
* RMSE
* OLS Regression Analysis

---

# 📂 Project Structure

```text
linear-regression-in-python/
│
├── simple-linear-regression/
│   └── Simple_Linear_Regression.ipynb
│
├── multiple-linear-regression/
│   └── multiple_linear_regression.ipynb
│
├── polynomial-regression/
│   ├── Polynomial_Regression.ipynb
│   └── polynomial_pipeline.ipynb
│
├── data/
│   ├── height-weight.csv
│   └── economic_index.csv
│
├── README.md
└── requirements.txt
```

---

# 1. Simple Linear Regression

### Dataset

The Simple Linear Regression notebook uses a **height-weight dataset** containing:

* `Weight` — Independent variable
* `Height` — Dependent variable

The objective is to understand whether **Weight can be used to predict Height**.

### Workflow

The notebook covers:

1. Loading the dataset
2. Understanding the dataset
3. Exploratory analysis
4. Separating independent and dependent variables
5. Train/Test Split
6. Feature scaling
7. Training a `LinearRegression` model
8. Extracting coefficient and intercept
9. Making predictions
10. Evaluating the model

### Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)
```

The fitted model produced:

* **Coefficient:** approximately `17.30`
* **Intercept:** approximately `156.47`

### Evaluation

The notebook evaluates the model using:

| Metric      | Result |
| ----------- | -----: |
| MSE         | 114.84 |
| MAE         |   9.67 |
| RMSE        |  10.72 |
| R² Score    |  0.736 |
| Adjusted R² |  0.670 |

The notebook also demonstrates residual analysis and OLS regression.

---

# 2. Multiple Linear Regression

### Dataset

The Multiple Linear Regression notebook uses an **economic index dataset** containing variables such as:

* `year`
* `month`
* `interest_rate`
* `unemployment_rate`
* `index_price`

The regression model uses multiple independent variables to predict:

**`index_price`**

### Exploratory Data Analysis

The notebook explores relationships between the variables using:

* Pair plots
* Scatter plots
* Correlation analysis

The correlation matrix shows strong relationships between the economic variables and `index_price`.

For example:

* Interest rate vs. index price: approximately **0.936**
* Unemployment rate vs. index price: approximately **-0.922**

### Workflow

The notebook covers:

1. Loading the economic dataset
2. Selecting independent and dependent variables
3. Exploratory Data Analysis
4. Correlation analysis
5. Train/Test Split
6. Linear Regression
7. Cross-validation
8. Prediction
9. Error analysis
10. R² Score
11. Adjusted R² Score
12. Residual analysis
13. OLS regression

### Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)
```

The model uses multiple independent variables to predict `index_price`.

### Evaluation

The held-out test-set results recorded in the notebook are:

| Metric      |  Result |
| ----------- | ------: |
| MSE         | 5793.76 |
| MAE         |   59.94 |
| RMSE        |   76.12 |
| R² Score    |   0.828 |
| Adjusted R² |   0.713 |

The notebook also demonstrates **3-fold cross-validation** using Mean Squared Error.

---

# 3. Polynomial Regression

Polynomial Regression is useful when the relationship between the independent and dependent variables is **non-linear** and a straight regression line does not adequately represent the data.

This section contains two notebooks demonstrating this concept.

---

## 3.1 Polynomial Regression on Non-Linear Data

The notebook first applies standard Linear Regression to a non-linear dataset.

### Linear Regression Result

The initial Linear Regression model achieved:

**R² Score: 0.740**

The notebook then visualises the fitted line and demonstrates that a straight line does not adequately capture the underlying non-linear relationship.

### Polynomial Features

Polynomial features are then created using:

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(
    degree=2,
    include_bias=True
)

x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)
```

A Linear Regression model is then trained using these transformed features.

### Polynomial Regression Result

After applying degree-2 polynomial features:

**R² Score: 0.909**

This demonstrates how transforming the original feature into polynomial features can allow a linear regression model to capture a non-linear relationship.

---

# 4. Polynomial Regression with Pipeline

The second polynomial regression notebook goes one step further by demonstrating how **PolynomialFeatures and LinearRegression can be combined using a Scikit-learn Pipeline**.

### Initial Linear Regression

The initial linear regression model achieved:

**R² Score: 0.563**

### Degree-2 Polynomial Regression

After applying polynomial features with degree 2:

**R² Score: 0.850**

The notebook also visualises the resulting regression curve.

### Pipeline

The workflow is implemented using:

```python
from sklearn.pipeline import Pipeline

poly_regression = Pipeline([
    ('Poly Features', PolynomialFeatures(degree=degree)),
    ('linear regression', LinearRegression())
])
```

A reusable function is also created to experiment with different polynomial degrees.

This demonstrates an important Machine Learning concept:

> **A pipeline can combine preprocessing/feature transformation and model training into a single workflow.**

---

# 📊 Regression Techniques Covered

| Technique                        | Dataset / Example | Main Concept                               |
| -------------------------------- | ----------------- | ------------------------------------------ |
| Simple Linear Regression         | Height-Weight     | One independent variable                   |
| Multiple Linear Regression       | Economic Index    | Multiple independent variables             |
| Polynomial Regression            | Non-linear data   | Capturing non-linear relationships         |
| Polynomial Regression + Pipeline | Non-linear data   | Combining feature transformation and model |

---

# 📈 Model Evaluation Metrics

The project demonstrates several commonly used regression evaluation metrics.

### Mean Absolute Error — MAE

Measures the average absolute difference between actual and predicted values.

A lower MAE indicates smaller prediction errors.

### Mean Squared Error — MSE

Calculates the average squared prediction error.

Larger errors receive greater weight because the errors are squared.

### Root Mean Squared Error — RMSE

RMSE is the square root of MSE and expresses the error in the same units as the target variable.

### R² Score

R² measures how much of the variation in the dependent variable is explained by the regression model.

### Adjusted R²

Adjusted R² accounts for the number of predictors included in the model and can be useful when working with multiple regression.

---

# 🔬 Statistical Analysis

The notebooks also introduce **Ordinary Least Squares (OLS)** regression using `statsmodels`.

This provides additional statistical information such as:

* Regression coefficients
* Standard errors
* t-statistics
* p-values
* Confidence intervals
* R²
* Adjusted R²
* F-statistic

Example:

```python
import statsmodels.api as sm

model_ols = sm.OLS(y_train, x_train)
results = model_ols.fit()

print(results.summary())
```

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Statsmodels
* Jupyter Notebook

---

# 🧠 Key Learning Outcomes

Through these notebooks, I developed practical understanding of:

* How Linear Regression works
* Difference between simple and multiple regression
* Independent vs. dependent variables
* Train/Test splitting
* Feature scaling
* Model coefficients and intercept
* Making predictions
* Regression evaluation metrics
* Cross-validation
* Residual analysis
* R² and Adjusted R²
* Understanding non-linear relationships
* Polynomial feature transformation
* Polynomial Regression
* Scikit-learn Pipelines
* OLS regression and statistical interpretation

---

# 🚀 Future Improvements

Possible improvements to this project include:

* Hyperparameter tuning for polynomial degree
* Comparing different polynomial degrees systematically
* More extensive cross-validation
* Residual diagnostics
* Feature selection
* Regularisation techniques such as Ridge and Lasso Regression
* Improved visualisation of model performance
* Building an interactive prediction application using Flask or Streamlit

---

# 📌 Project Purpose

This repository is part of my ongoing **Data Science and Machine Learning learning journey**.

The main objective is to strengthen my understanding of regression algorithms by implementing them from end to end rather than only studying the theory.

The progression in this repository is:

**Simple Regression → Multiple Regression → Non-Linear Relationships → Polynomial Regression → Pipelines**

---

# 👨‍💻 Author

**Ashish Singh**

MSc Computer Science | Aspiring Data Scientist

Currently building practical projects and strengthening my skills in:

**Python | Data Analysis | Statistics | Machine Learning | SQL**

---

## ⭐ If you find this project useful

Feel free to explore the notebooks and datasets, and check out my other Data Science projects on GitHub.
