# Machine Learning Projects: Classification & Regression

Two end-to-end machine learning projects built in Python with scikit-learn and XGBoost, covering one **classification** problem and one **regression** problem.

| Project | Problem type | Target | Best model | Key test result |
|---|---|---|---|---|
| [Travel Package Purchase Prediction](#1-travel-package-purchase-prediction) | Binary classification | `ProdTaken` (bought the package or not) | Gradient Boosting (tuned) | F1 = 0.83, Precision = 0.97, ROC-AUC = 0.86 |
| [Car Price Prediction](#2-car-price-prediction) | Regression | `selling_price` | Random Forest | R² = 0.931 |

---

## Repository Structure

```
├── Travel_Package_Purchase_Prediction.ipynb
├── Car_Price_Prediction.ipynb
├── Travel.csv                    # dataset for project 1 (add to repo)
├── cardekho_imputated.csv        # dataset for project 2 (add to repo)
└── README.md
```

## Tech Stack

- **Language:** Python
- **Data handling and visualisation:** pandas, NumPy, Matplotlib, Seaborn, Plotly
- **Modelling:** scikit-learn, XGBoost
- **Techniques:** missing value imputation, feature engineering, `ColumnTransformer` pipelines (scaling and encoding), train/test split, model comparison, hyperparameter tuning with `RandomizedSearchCV`

---

## 1. Travel Package Purchase Prediction

**Notebook:** `Travel_Package_Purchase_Prediction.ipynb`

### Objective
Predict whether a customer will purchase a travel package, based on customer demographics and details of the sales interaction. A model like this helps a travel company focus its marketing and sales effort on customers who are most likely to buy.

### Dataset
- 4,888 customer records, 20 columns (`Travel.csv`)
- Target: `ProdTaken` (1 = purchased, 0 = did not purchase)
- Roughly 19% of customers purchased, so the classes are imbalanced
- Features include `Age`, `TypeofContact`, `CityTier`, `DurationOfPitch`, `Occupation`, `Gender`, `ProductPitched`, `PreferredPropertyStar`, `MaritalStatus`, `NumberOfTrips`, `Passport`, `PitchSatisfactionScore`, `OwnCar`, `Designation`, `MonthlyIncome`

### Workflow
1. **Missing values:** 8 columns had nulls (0.5% to 5%). Numerical columns with a wide spread (`Age`, `DurationOfPitch`, `MonthlyIncome`) were imputed with the **median**; discrete and categorical columns were imputed with the **mode**.
2. **Feature engineering:** combined `NumberOfPersonVisiting` and `NumberOfChildrenVisiting` into a single `TotalVisiting` feature.
3. **Feature selection:** dropped the `CustomerID` identifier.
4. **Preprocessing:** `StandardScaler` for numerical features and `OneHotEncoder(drop='first')` for categorical features, combined in a `ColumnTransformer`.
5. **Split:** 80/20 train/test split (`random_state=42`).
6. **Modelling:** compared AdaBoost, Gradient Boosting, XGBoost and Random Forest.
7. **Tuning:** `RandomizedSearchCV` (5-fold CV, 100 iterations) on Random Forest and Gradient Boosting.
8. **Evaluation:** F1, recall, precision, accuracy, confusion matrix, ROC-AUC and ROC curve.

### Results (test set)

**Baseline models**

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| AdaBoost | 0.836 | 0.772 | 0.230 | 0.355 |
| Gradient Boosting | 0.856 | 0.755 | 0.387 | 0.512 |
| XGBoost | 0.933 | 0.919 | 0.717 | 0.806 |
| Random Forest | 0.919 | 0.959 | 0.613 | 0.748 |

**After hyperparameter tuning**

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Gradient Boosting (tuned) | 0.966 | 0.733 | **0.833** | **0.863** |
| Random Forest (tuned) | 0.967 | 0.607 | 0.746 | 0.801 |

### Key Takeaways
- Boosting and bagging models with deeper trees clearly outperformed the shallow AdaBoost baseline, especially on recall.
- Tuned **Gradient Boosting** gave the best balance of precision and recall (F1 0.83).
- Because the target is imbalanced, accuracy alone is misleading, so F1, recall and ROC-AUC were used as the main metrics.
- The tuned models score 1.0 on the training set against a lower test score, which indicates overfitting. Regularisation (a lower `max_depth`, a smaller learning rate) and cross-validated evaluation would be the next step.

---

## 2. Car Price Prediction

**Notebook:** `Car_Price_Prediction.ipynb`

### Objective
Predict the resale price of a used car from its specifications and usage history, using listings from the CarDekho used-car marketplace.

### Dataset
- Used-car listings (`cardekho_imputated.csv`) with no missing values
- Target: `selling_price`
- Features: `model`, `vehicle_age`, `km_driven`, `seller_type`, `fuel_type`, `transmission_type`, `mileage`, `engine`, `max_power`, `seats`

### Workflow
1. **Data cleaning:** dropped the index column `Unnamed: 0`, confirmed no null values.
2. **Feature analysis:** separated numerical, categorical, discrete and continuous features and inspected the distributions of `vehicle_age` and `seats`.
3. **Feature selection:** dropped `car_name` and `brand`, which are redundant with `model`.
4. **Encoding:**
   - `model` (high cardinality) with `LabelEncoder`
   - `seller_type`, `fuel_type`, `transmission_type` with `OneHotEncoder(drop='first')`
   - numerical features with `StandardScaler`
5. **Split:** 80/20 train/test split (`random_state=42`).
6. **Modelling:** compared Decision Tree, Random Forest, Gradient Boosting, K-Nearest Neighbours and XGBoost regressors.
7. **Tuning:** `RandomizedSearchCV` (5-fold CV, 100 iterations) on the Decision Tree.
8. **Evaluation:** R², MAE, MSE and RMSE on both train and test sets.

### Results (test set)

| Model | R² | MAE | RMSE |
|---|---|---|---|
| **Random Forest** | **0.9310** | 101,616 | 227,908 |
| Gradient Boosting | 0.9139 | 126,038 | 254,651 |
| K-Neighbours | 0.9116 | 113,064 | 257,928 |
| Decision Tree | 0.8810 | 124,046 | 299,358 |
| XGBoost | 0.8209 | 102,948 | 367,142 |
| Decision Tree (tuned) | 0.7989 | 170,957 | 389,088 |

### Key Takeaways
- **Random Forest** gave the best generalisation (test R² 0.931, lowest MAE and RMSE among the compared models).
- The untuned Decision Tree overfit heavily (train R² 0.9995 against test R² 0.881). Tuning reduced the train/test gap (0.875 vs 0.799) but at the cost of overall accuracy, so a shallow tree is not the right model for this problem.
- Ensemble methods handled the mix of categorical and numerical features better than a single tree.

---

## Future Improvements
- Handle class imbalance in the travel dataset (SMOTE, class weights, threshold tuning)
- Tune more models (XGBoost, Gradient Boosting for the car data) and use `GridSearchCV` around the best random search results
- Apply a log transform to `selling_price` to reduce the effect of expensive outliers
- Add feature importance and SHAP plots for interpretability
- Package the best models into a simple web app (Streamlit or Flask)

## Author
**Ashish Kumar Singh**
MSc Computer Science, Birmingham City University
Focus areas: Python, data analysis, statistics, machine learning
