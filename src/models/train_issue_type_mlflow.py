# src/models/train_issue_type_mlflow.py

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn  # enables sklearn helpers [web:138]


DATA_PATH = Path("data/processed/react_issues.parquet")
MODEL_DIR = Path("models/issue_type_baseline")
MODEL_PATH = MODEL_DIR / "model.joblib"
EXPERIMENT_NAME = "react_issue_type_baseline"


def load_data(path: Path) -> pd.DataFrame:
    assert path.exists(), f"Processed dataset not found at {path}"
    df = pd.read_parquet(path)
    df = df[df["issue_type"].notna()].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.len() > 0]
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"])
    df = df.sort_values("created_at").reset_index(drop=True)
    return df


def time_based_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()
    print(f"Total samples: {n}")
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    return df_train, df_val, df_test


def build_pipeline() -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
    )
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
    )
    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def eval_split(model: Pipeline, X, y, split_name: str):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")
    print(f"\n=== {split_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(y, preds))
    # Log metrics to MLflow [web:133][web:134]
    mlflow.log_metric(f"{split_name}_accuracy", acc)
    mlflow.log_metric(f"{split_name}_macro_f1", macro_f1)


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(DATA_PATH)
    print(df["issue_type"].value_counts())

    df_train, df_val, df_test = time_based_split(df)
    X_train, y_train = df_train["text"], df_train["issue_type"]
    X_val, y_val = df_val["text"], df_val["issue_type"]
    X_test, y_test = df_test["text"], df_test["issue_type"]

    params = {
        "max_features": 8000,
        "ngram_range": "(1,2)",
        "min_df": 2,
        "clf_max_iter": 1000,
        "clf_class_weight": "balanced",
    }

    with mlflow.start_run(run_name="tfidf_logreg_baseline"):  # [web:133][web:134]
        # Log hyperparameters
        mlflow.log_params(params)

        model = build_pipeline()
        print("\nTraining model...")
        model.fit(X_train, y_train)

        # Evaluate and log metrics
        eval_split(model, X_train, y_train, "train")
        eval_split(model, X_val, y_val, "val")
        eval_split(model, X_test, y_test, "test")

        # Log model using MLflow's sklearn helper [web:138][web:144]
        input_example_df = pd.DataFrame({"text": X_train.iloc[:1].tolist()})
        signature = mlflow.models.infer_signature(
            input_example_df, model.predict(X_train)
        )  # [web:141]
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example_df,
        )

        # Also save locally for FastAPI
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"\nSaved local model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
