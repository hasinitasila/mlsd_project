import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow
import mlflow.sklearn

# ── 1. Load & save raw data ──────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df.to_csv("data/iris.csv", index=False)
print("✅ Data saved → data/iris.csv")

# ── 2. Preprocess ────────────────────────────────────────────────────────────
X = df[iris.feature_names].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── 3. MLflow experiment setup ───────────────────────────────────────────────
mlflow.set_experiment("iris-mlp-classifier")

with mlflow.start_run():

    # ── 4. Define hyperparameters ────────────────────────────────────────────
    params = {
        "hidden_layer_sizes": (128, 64, 32),
        "activation":         "relu",
        "solver":             "adam",
        "max_iter":           2000,
        "random_state":       42,
        "learning_rate_init": 0.001,
    }

    # ── 5. Log parameters to MLflow ──────────────────────────────────────────
    mlflow.log_param("hidden_layers",       str(params["hidden_layer_sizes"]))
    mlflow.log_param("activation",          params["activation"])
    mlflow.log_param("solver",              params["solver"])
    mlflow.log_param("max_iter",            params["max_iter"])
    mlflow.log_param("learning_rate_init",  params["learning_rate_init"])
    mlflow.log_param("test_size",           0.2)

    # ── 6. Train ─────────────────────────────────────────────────────────────
    model = MLPClassifier(**params)
    model.fit(X_train, y_train)

    # ── 7. Evaluate ──────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=iris.target_names)

    print(f"\n📊 Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)

    # ── 8. Log metrics to MLflow ─────────────────────────────────────────────
    mlflow.log_metric("accuracy",       round(acc, 4))
    mlflow.log_metric("train_size",     len(y_train))
    mlflow.log_metric("test_size",      len(y_test))
    mlflow.log_metric("n_iter",         model.n_iter_)
    mlflow.log_metric("loss",           round(model.loss_, 6))

    # ── 9. Log model to MLflow (registered as artifact) ──────────────────────
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="mlp-model",
        registered_model_name="iris-mlp-classifier",
    )

    # ── 10. Also save locally for Streamlit + Docker ─────────────────────────
    joblib.dump(model,  "model/mlp_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    # ── 11. Save metrics.json (used by DVC + Streamlit) ──────────────────────
    metrics = {
        "accuracy":      round(acc, 4),
        "test_size":     len(y_test),
        "train_size":    len(y_train),
        "hidden_layers": [128, 64, 32],
        "activation":    "relu",
        "target_names":  list(iris.target_names),
        "feature_names": list(iris.feature_names),
    }
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── 12. Log metrics.json as an MLflow artifact too ───────────────────────
    mlflow.log_artifact("model/metrics.json")

    print("\n✅ Model saved  → model/mlp_model.pkl")
    print("✅ Scaler saved → model/scaler.pkl")
    print("✅ Metrics saved → model/metrics.json")
    print("✅ MLflow run logged → run 'mlflow ui' to view")
