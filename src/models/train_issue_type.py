# src/models/train_issue_type.py

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/processed/react_issues.parquet")
MODEL_DIR = Path("models/issue_type_baseline")
MODEL_PATH = MODEL_DIR / "model.joblib"


def load_data(path: Path) -> pd.DataFrame:
    assert path.exists(), f"Processed dataset not found at {path}"
    df = pd.read_parquet(path)

    # Keep only rows with non-null issue_type and non-empty text
    df = df[df["issue_type"].notna()].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.len() > 0]

    # Ensure created_at is datetime for time-based split
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"])

    df = df.sort_values("created_at").reset_index(drop=True)
    return df


def time_based_split(
    df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train/val/test in chronological order.
    """
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
    """
    Create an sklearn Pipeline: TF-IDF vectorizer + Logistic Regression classifier.
    """
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

    pipe = Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("clf", clf),
        ]
    )
    return pipe


def evaluate_model(
    model: Pipeline, X, y, split_name: str
) -> None:
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")

    print(f"\n=== {split_name} performance ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y, preds))


def main() -> None:
    # 1. Load data
    df = load_data(DATA_PATH)
    print(df["issue_type"].value_counts())

    # 2. Time-based split
    df_train, df_val, df_test = time_based_split(df)

    X_train, y_train = df_train["text"], df_train["issue_type"]
    X_val, y_val = df_val["text"], df_val["issue_type"]
    X_test, y_test = df_test["text"], df_test["issue_type"]

    # 3. Build model pipeline
    model = build_pipeline()

    # 4. Train on train set
    print("\nTraining model...")
    model.fit(X_train, y_train)

    # 5. Evaluate on train/val/test
    evaluate_model(model, X_train, y_train, "Train")
    evaluate_model(model, X_val, y_val, "Validation")
    evaluate_model(model, X_test, y_test, "Test")

    # 6. Save trained model pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
