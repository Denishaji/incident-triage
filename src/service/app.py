# src/service/app.py

import os
import joblib
from pathlib import Path
from typing import List, Literal
from fastapi import FastAPI
from pydantic import BaseModel

# --- ABSOLUTE PATH LOGIC ---
# This ensures the model is found on Render's Linux server
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "issue_type_baseline" / "model.joblib"

app = FastAPI(
    title="React Incident Triage API",
    description="Predicts issue_type (bug/feature/question) from GitHub issue text.",
    version="0.1.0",
)

class IssueRequest(BaseModel):
    title: str
    body: str

class IssuePrediction(BaseModel):
    issue_type: Literal["bug", "feature", "question"]
    probabilities: dict

model = None

@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict_issue_type", response_model=IssuePrediction)
def predict_issue_type(issue: IssueRequest):
    if model is None:
        return {"error": "Model not loaded."}
    text = f"{issue.title.strip()}\n\n{issue.body.strip()}"
    preds = model.predict([text])
    probs = model.predict_proba([text])[0]
    classes = list(model.classes_)
    prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
    return IssuePrediction(issue_type=preds[0], probabilities=prob_dict)

# This part is for Render to know which port to use
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)