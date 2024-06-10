import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("TelcoChurn.csv")
    return data

# Preprocess the data
def preprocess_data(data):
    data.drop('customerID', axis=1, inplace=True)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data.dropna(subset=['TotalCharges'], inplace=True)
    categorical_cols = data.select_dtypes(include=['object']).columns
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    X = data_encoded.drop('Churn_Yes', axis=1)
    y = data_encoded['Churn_Yes']
    return X, y

# Feature Engineering
def feature_engineering(data):
    data['AvgMonthlyCharges'] = data['TotalCharges'] / data['tenure']
    
    def tenure_group(tenure):
        if tenure <= 12:
            return '0-12'
        elif tenure <= 24:
            return '13-24'
        elif tenure <= 36:
            return '25-36'
        elif tenure <= 48:
            return '37-48'
        elif tenure <= 60:
            return '49-60'
        else:
            return '61+'

    data['tenure_group'] = data['tenure'].apply(tenure_group)

    data['MultipleServices'] = (
        (data['PhoneService'] == 'Yes').astype(int) +
        (data['InternetService'] != 'No').astype(int)
    )
    data['HasMultipleServices'] = (data['MultipleServices'] > 1).astype(int)
    data.drop('MultipleServices', axis=1, inplace=True)
    
    categorical_cols = data.select_dtypes(include=['object']).columns
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    X = data_encoded.drop('Churn_Yes', axis=1)
    y = data_encoded['Churn_Yes']
    return X, y

# Load and preprocess the data
data = load_data()
X, y = preprocess_data(data)
X, y = feature_engineering(data)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
log_reg = LogisticRegression(max_iter=1000)
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
gradient_boosting = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the models
log_reg.fit(X_train, y_train)
random_forest.fit(X_train, y_train)
gradient_boosting.fit(X_train, y_train)

# Evaluate the models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1, y_pred

log_reg_metrics = evaluate_model(log_reg, X_test, y_test)
rf_metrics = evaluate_model(random_forest, X_test, y_test)
gb_metrics = evaluate_model(gradient_boosting, X_test, y_test)

# App Introduction
st.title("Telco Customer Churn Prediction")
st.write("""
## Developed by Abhishek Kumar Singh
### Course: BTech (ML & CSE)

#### Project Introduction:
This project aims to predict customer churn in a Telco company using various machine learning models. 
We will evaluate the performance of Logistic Regression, Random Forest, and Gradient Boosting models.

#### Dataset Introduction:
The dataset contains information about Telco customers, including demographics, account information, and service details.
""")

# Model Selection
st.sidebar.title("Model Selection")
model_option = st.sidebar.selectbox(
    'Select a model to evaluate:',
    ('Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Show All Models')
)

# Function to display evaluation metrics
def display_metrics(metrics, model_name):
    st.write(f"### {model_name} Performance Metrics")
    st.write(f"Accuracy: {metrics[0]:.2f}")
    st.write(f"Precision: {metrics[1]:.2f}")
    st.write(f"Recall: {metrics[2]:.2f}")
    st.write(f"F1-Score: {metrics[3]:.2f}")

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred, model_name):
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'], ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title(f'Confusion Matrix - {model_name}')
    st.pyplot(fig)

# Function to plot ROC curve
def plot_roc_curve(model, X_test, y_test, model_name):
    fig, ax = plt.subplots()
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc="lower right")
    st.pyplot(fig)

# Function to plot radar chart
def plot_radar_chart(metrics, model_name):
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = list(metrics[:4])
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(f'Radar Chart - {model_name}', size=20, color='blue', y=1.1)
    st.pyplot(fig)

# Display metrics and plots based on selected model
if model_option == 'Logistic Regression':
    display_metrics(log_reg_metrics, 'Logistic Regression')
    plot_confusion_matrix(y_test, log_reg_metrics[4], 'Logistic Regression')
    plot_roc_curve(log_reg, X_test, y_test, 'Logistic Regression')
    plot_radar_chart(log_reg_metrics, 'Logistic Regression')

elif model_option == 'Random Forest':
    display_metrics(rf_metrics, 'Random Forest')
    plot_confusion_matrix(y_test, rf_metrics[4], 'Random Forest')
    plot_roc_curve(random_forest, X_test, y_test, 'Random Forest')
    plot_radar_chart(rf_metrics, 'Random Forest')

elif model_option == 'Gradient Boosting':
    display_metrics(gb_metrics, 'Gradient Boosting')
    plot_confusion_matrix(y_test, gb_metrics[4], 'Gradient Boosting')
    plot_roc_curve(gradient_boosting, X_test, y_test, 'Gradient Boosting')
    plot_radar_chart(gb_metrics, 'Gradient Boosting')

elif model_option == 'Show All Models':
    st.write("### Logistic Regression Performance Metrics")
    display_metrics(log_reg_metrics, 'Logistic Regression')
    plot_confusion_matrix(y_test, log_reg_metrics[4], 'Logistic Regression')
    plot_roc_curve(log_reg, X_test, y_test, 'Logistic Regression')
    plot_radar_chart(log_reg_metrics, 'Logistic Regression')

    st.write("### Random Forest Performance Metrics")
    display_metrics(rf_metrics, 'Random Forest')
    plot_confusion_matrix(y_test, rf_metrics[4], 'Random Forest')
    plot_roc_curve(random_forest, X_test, y_test, 'Random Forest')
    plot_radar_chart(rf_metrics, 'Random Forest')

    st.write("### Gradient Boosting Performance Metrics")
    display_metrics(gb_metrics, 'Gradient Boosting')
    plot_confusion_matrix(y_test, gb_metrics[4], 'Gradient Boosting')
    plot_roc_curve(gradient_boosting, X_test, y_test, 'Gradient Boosting')
    plot_radar_chart(gb_metrics, 'Gradient Boosting')
