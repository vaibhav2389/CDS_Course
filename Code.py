# ml_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Exploratory Data Analysis (EDA)
def eda(data):
    print("First 5 rows of the data:")
    print(data.head())

    print("\nData Information:")
    print(data.info())

    print("\nDescriptive Statistics:")
    print(data.describe())

    # Plot pairplot
    sns.pairplot(data)
    plt.show()

    # Plot correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

# Feature Engineering
def feature_engineering(data):
    # Example: Encode categorical features
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    return data, label_encoders

# Model Building and Evaluation
def model_building(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100)
    }

    # Evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        if name in ['Linear Regression']:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name] = {'MSE': mse, 'R2 Score': r2}
        else:
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred)
            results[name] = {'Accuracy': accuracy, 'Confusion Matrix': cm, 'Classification Report': cr}

    return results

# Visualization
def visualize_results(results):
    # Convert results to a DataFrame
    results_df = pd.DataFrame(results).T

    # Plot accuracy or other metrics
    fig = go.Figure()
    for metric in results_df.columns:
        fig.add_trace(go.Bar(x=results_df.index, y=results_df[metric], name=metric))

    fig.update_layout(barmode='group', title='Model Performance')
    fig.show()

# Main function
if __name__ == "__main__":
    # Load your dataset
    file_path = 'your_dataset.csv'  # Replace with your actual data file path
    data = load_data(file_path)

    # Perform EDA
    eda(data)

    # Feature Engineering
    data, label_encoders = feature_engineering(data)

    # Build and evaluate models
    target_column = 'target'  # Replace with your actual target column
    results = model_building(data, target_column)

    # Visualize results
    visualize_results(results)
