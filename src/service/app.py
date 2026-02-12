# src/service/app.py

import os
import joblib
from pathlib import Path
from typing import List, Literal
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

# --- ROBUST PATH LOGIC ---
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

# --- UPDATED INTERACTIVE HOME PAGE ---
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>Incident Triage AI</title>
            <style>
                body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; background: #f6f8fa; color: #24292e; }
                .container { border: 1px solid #e1e4e8; border-radius: 10px; padding: 30px; background: #fff; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
                h1 { border-bottom: 2px solid #0366d6; padding-bottom: 10px; }
                label { display: block; margin: 15px 0 5px; font-weight: bold; }
                input, textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 6px; font-size: 16px; }
                textarea { height: 100px; resize: vertical; }
                button { background: #28a745; color: white; padding: 12px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; margin-top: 15px; font-weight: bold; width: 100%; }
                button:hover { background: #218838; }
                #result { margin-top: 20px; padding: 15px; border-radius: 6px; display: none; }
                .bug { background: #ffdce0; color: #86181d; border: 1px solid #cea0a5; }
                .feature { background: #dbedff; color: #005cc5; border: 1px solid #a2c1e8; }
                .question { background: #fff5b1; color: #735c0f; border: 1px solid #e2d57e; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Incident Triage AI</h1>
                <p>Enter a GitHub issue below to see the ML model predict the category.</p>
                
                <label>Issue Title</label>
                <input type="text" id="title" placeholder="e.g., App crashes on login screen">
                
                <label>Issue Description</label>
                <textarea id="body" placeholder="e.g., When I enter my password and click submit, the application closes immediately."></textarea>
                
                <button onclick="predict()">Analyze Issue Type</button>
                
                <div id="result"></div>
                
                <hr style="margin-top:30px; border:0; border-top:1px solid #eee;">
                <p style="font-size:0.9em; color:#666;">Technical Reviewer? <a href="/docs">View Swagger API Docs</a></p>
            </div>

            <script>
                async function predict() {
                    const title = document.getElementById('title').value;
                    const body = document.getElementById('body').value;
                    const resultDiv = document.getElementById('result');

                    if(!title || !body) { alert('Please fill in both fields'); return; }

                    resultDiv.style.display = 'block';
                    resultDiv.innerHTML = 'Analyzing...';
                    resultDiv.className = '';

                    const response = await fetch('/predict_issue_type', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ title, body })
                    });

                    const data = await response.json();
                    
                    resultDiv.className = data.issue_type;
                    resultDiv.innerHTML = `<strong>Predicted Type:</strong> ${data.issue_type.toUpperCase()}<br>
                                           <small>Confidence: ${(Math.max(...Object.values(data.probabilities)) * 100).toFixed(2)}%</small>`;
                }
            </script>
        </body>
    </html>
    """

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
    if model is None: return {"error": "Model not loaded"}
    text = f"{issue.title.strip()}\n\n{issue.body.strip()}"
    preds = model.predict([text])
    probs = model.predict_proba([text])[0]
    classes = list(model.classes_)
    prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
    return IssuePrediction(issue_type=preds[0], probabilities=prob_dict)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)