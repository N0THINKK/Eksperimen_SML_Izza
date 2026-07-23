import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

def load_and_preprocess():
    print("Loading Bank Marketing dataset...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
    import urllib.request, zipfile, io
    r = urllib.request.urlopen(url)
    z = zipfile.ZipFile(io.BytesIO(r.read()))
    df = pd.read_csv(z.open('bank.csv'), sep=';')

    print(f"Dataset loaded: {df.shape}")
    df.to_csv('bank_raw.csv', index=False)

    # Encode categorical columns
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols.remove('y')
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    df['y'] = (df['y'] == 'yes').astype(int)

    X = df.drop('y', axis=1)
    y = df['y']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs('bank_preprocessing', exist_ok=True)
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['target'] = y_train.values
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['target'] = y_test.values
    train_df.to_csv('bank_preprocessing/train.csv', index=False)
    test_df.to_csv('bank_preprocessing/test.csv', index=False)

    return X_train_scaled, X_test_scaled, y_train, y_test

def train():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    mlflow.set_experiment("bank-marketing-classification")
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="gradient-boosting-basic"):
        model = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1,
            max_depth=5, random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("Training selesai. Cek MLflow UI: http://localhost:5000")

if __name__ == "__main__":
    train()
