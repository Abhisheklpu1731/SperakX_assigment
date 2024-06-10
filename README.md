# SperakX_assigment
# Telco Churn Prediction Project

This project predicts customer churn in a telecommunications company using machine learning techniques. It involves data loading, preprocessing, exploratory data analysis (EDA), feature engineering, model building, and evaluation. Key steps include environment setup, data preprocessing, EDA, feature engineering, model training, and visualization of results.

## Overview

The project aims to identify customers at risk of churning, enabling targeted retention strategies and minimizing revenue loss. It utilizes various machine learning models to predict churn based on customer demographics, services, and contract details.

## Files Uploaded

- Abhishek_cv.pdf: Resume of Abhishek Kumar Singh.
- README.md: Overview and description of files in the project.
- SpeakX_assigment_word.pdf: Assignment document.
- TelcoChurn.csv: Dataset containing customer information and churn status.
- speakx.py: Python script file.
- speakx_assignment.ipynb: Jupyter Notebook containing the project code.
- speakx_assignment.py: Python script file (same as speakx.py).
- web.py: Python script for web application (not described in detail).

## Setup and Libraries

The project requires the following libraries:
- Pandas and NumPy for data manipulation
- Matplotlib and Seaborn for visualization
- Scikit-learn for machine learning models

## Data Loading and Preprocessing

- The dataset (TelcoChurn.csv) is loaded using Pandas.
- Irrelevant columns are dropped, and missing values are handled.
- Categorical variables are encoded for analysis.

## Exploratory Data Analysis (EDA)

- Various visualizations are created to understand the distribution and relationships of variables.
- EDA includes churn distribution, tenure distribution, churn by tenure, monthly charges, total charges, and contract type.

## Feature Engineering

- New features are created, including average monthly charges, tenure groups, and multiple services.
- Categorical variables are encoded for model training.

## Model Building

- The dataset is split into training and testing sets.
- Machine learning models such as Logistic Regression, Random Forest, and Gradient Boosting are trained and evaluated.
- Model performance metrics like accuracy, precision, recall, and F1-score are calculated.

## Results

- Confusion matrices and ROC curves are plotted to visualize model performance.
- Metrics for each model, including accuracy, precision, recall, and F1-score, are displayed.
- Radar charts provide a comparative view of model performance across different metrics.

## Conclusion

This project enables the telecommunications company to proactively identify customers likely to churn, facilitating targeted retention efforts and enhancing customer satisfaction and revenue retention.

