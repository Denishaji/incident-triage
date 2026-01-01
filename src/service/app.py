# src/service/app.py

from pathlib import Path
from typing import List, Literal

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path("models/issue_type_baseline/model.joblib")

app = FastAPI(
    title="React Incident Triage API",
    description="Predicts issue_type (bug/feature/question) from GitHub issue text.",
    version="0.1.0",
)

# Pydantic models for request/response [web:165]
class IssueRequest(BaseModel):
    title: str
    body: str


class IssuePrediction(BaseModel):
    issue_type: Literal["bug", "feature", "question"]
    probabilities: dict


# Load model at startup
@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Train the model first.")
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict_issue_type", response_model=IssuePrediction)
def predict_issue_type(issue: IssueRequest):
    """
    Predict issue_type from title + body text.
    """
    text = f"{issue.title.strip()}\n\n{issue.body.strip()}"
    preds = model.predict([text])
    probs = model.predict_proba([text])[0]

    classes = list(model.classes_)
    prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}

    return IssuePrediction(issue_type=preds[0], probabilities=prob_dict)
