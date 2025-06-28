# Algerian Forest Fire – Linear Regression Modeling

This project uses the Algerian Forest Fires dataset from the UCI Machine Learning Repository to predict fire risk using linear regression. The dataset includes temperature, humidity, wind, rain, and Fire Weather Index (FWI) components recorded across two Algerian regions. The project includes EDA, feature engineering, model training, and evaluation using R² and RMSE.

---

## Objective

The goal is to develop and evaluate regression models that can estimate the Fire Weather Index (FWI) using climate-related and calculated features. This includes:

- Data preprocessing and cleaning  
- Feature distribution analysis and outlier handling  
- Correlation and multicollinearity checks  
- Application of Linear Regression and its regularized variants (Ridge, Lasso)  
- Cross-validation and hyperparameter tuning  
- Model evaluation using multiple metrics  
- Saving the best model using pickle for future inference  

---

## Dataset

The dataset includes daily weather observations and FWI components from two regions in Algeria (Bejaia and Sidi Bel-abbes), collected between **June and September 2012**. It contains 244 instances and 12 attributes, including temperature, wind speed, humidity, and indices from the Canadian FWI system.

---

## Setup

### Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- statsmodels

To install dependencies:
```bash
pip install -r requirements.txt
```


## Key Steps

### 1. Data Cleaning & Transformation
- Applied log transformation to skewed features
- Analyzed and preserved outliers for interpretability
- Scaled all numerical features using standardization

### 2. Exploratory Data Analysis
- Plotted histograms and boxplots for numerical columns
- Used correlation matrix to detect feature interactions
- Removed highly correlated features (threshold > 0.90) for Linear Regression

### 3. Model Development
- Trained baseline Linear Regression
- Applied Ridge and Lasso with:
  - `RidgeCV` and `LassoCV` for quick alpha selection
  - `GridSearchCV` for advanced hyperparameter tuning

### 4. Evaluation
- Evaluation metrics used: `MAE`, `RMSE`, `R²`
- Ridge (tuned via `GridSearchCV`) gave the best overall performance
- Included `MAE` as a key metric due to the presence of outliers

### 5. Model Saving
- Best model (`Tuned Ridge`) saved as a `.pkl` file using `pickle`

---

