import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import os

# Setup MLflow
mlflow.set_experiment("iris-classification")

def load_and_preprocess():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Simpan data preprocessing
    os.makedirs('iris_preprocessing', exist_ok=True)
    train_df = pd.DataFrame(X_train_scaled, columns=iris.feature_names)
    train_df['target'] = y_train.values
    test_df = pd.DataFrame(X_test_scaled, columns=iris.feature_names)
    test_df['target'] = y_test.values
    train_df.to_csv('iris_preprocessing/train.csv', index=False)
    test_df.to_csv('iris_preprocessing/test.csv', index=False)

    return X_train_scaled, X_test_scaled, y_train, y_test

def train():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # MLflow autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="logistic-regression-basic"):
        model = LogisticRegression(random_state=42, max_iter=200)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("Training selesai. Cek MLflow UI: http://localhost:5000")

if __name__ == "__main__":
    train()
