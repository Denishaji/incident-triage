# src/data/prepare_issues.py

from pathlib import Path
from typing import List, Optional

import pandas as pd


RAW_PATH = Path("data/raw/react_issues.json")
PROCESSED_DIR = Path("data/processed")
PROCESSED_PATH = PROCESSED_DIR / "react_issues.parquet"


def extract_issue_type(labels: List[str]) -> Optional[str]:
    """
    From a list of label names like ['Type: Bug', 'Status: Unconfirmed'],
    return a simple issue_type string such as 'bug', 'feature', or None
    if no suitable type label is present.
    """
    if not labels:
        return None

    # Normalize labels to lower-case for comparison
    lower_labels = [lbl.lower() for lbl in labels]

    # Example: 'Type: Bug' or 'bug'
    for lbl in lower_labels:
        if "type: bug" in lbl or lbl.strip() == "bug":
            return "bug"
        if "type: regression" in lbl:
            return "bug"  # treat regressions as bugs
        if "type: feature" in lbl or "enhancement" in lbl:
            return "feature"
        if "type: documentation" in lbl or "docs" in lbl:
            return "docs"
        if "type: question" in lbl or "question" in lbl:
            return "question"

    # Fallback: no recognized type
    return None


def extract_status(labels: List[str]) -> Optional[str]:
    """
    Extract a simple status from labels like 'Status: Unconfirmed'.
    """
    if not labels:
        return None

    for lbl in labels:
        if lbl.lower().startswith("status:"):
            # e.g. 'Status: Unconfirmed' -> 'unconfirmed'
            parts = lbl.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip().lower()
    return None


def build_text(title: str, body: str) -> str:
    """
    Combine title and body into a single text field.
    """
    title = title or ""
    body = body or ""
    return (title.strip() + "\n\n" + body.strip()).strip()


def prepare() -> pd.DataFrame:
    """
    Load raw issues, derive text and label columns, and return a cleaned DataFrame.
    """
    assert RAW_PATH.exists(), f"Raw file not found: {RAW_PATH}"

    df = pd.read_json(RAW_PATH)

    # Ensure labels is always a list (handle nulls)
    df["labels"] = df["labels"].apply(lambda x: x if isinstance(x, list) else [])

    # Derive target columns
    df["issue_type"] = df["labels"].apply(extract_issue_type)
    df["status"] = df["labels"].apply(extract_status)

    # Build combined text
    df["text"] = df.apply(lambda row: build_text(row.get("title", ""), row.get("body", "")), axis=1)

    # Basic filtering: non-empty text and a known issue_type
    df["text_length"] = df["text"].str.len()
    df_clean = df[(df["text_length"] > 0) & df["issue_type"].notna()].copy()

    # Optional: drop helper columns
    df_clean = df_clean.drop(columns=["text_length"])

    return df_clean


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_clean = prepare()

    # Save as parquet (more efficient than CSV)
    df_clean.to_parquet(PROCESSED_PATH, index=False)

    print(f"Prepared dataset with {len(df_clean)} rows.")
    print(f"Saved to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
