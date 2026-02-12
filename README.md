# Intelligent Incident Triage System

An end-to-end ML pipeline that automatically classifies GitHub issues by type (bug/feature/question) using real data from the `facebook/react` repository. This project demonstrates production-grade ML engineering practices including data ingestion, experiment tracking, model versioning, and API deployment.

##  Project Overview

This system ingests real GitHub issues via REST API, trains a text classifier to predict issue types, tracks experiments with MLflow, and serves predictions through a FastAPI microservice.

**Key Features:**
- ✅ Automated data collection from GitHub REST API with authentication and pagination
- ✅ Clean ETL pipeline with data validation and preprocessing
- ✅ Time-based train/val/test splits for realistic evaluation
- ✅ Experiment tracking and model versioning with MLflow
- ✅ Production-ready REST API with FastAPI
- ✅ Structured codebase following ML engineering best practices

**Business Value:** Reduces manual triage effort by auto-classifying incoming issues, enabling faster routing to the correct team.
---

##  Architecture
```
GitHub Issues (REST API)
↓
Data Ingestion → data/raw/react_issues.json
↓
Preprocessing → data/processed/react_issues.parquet
↓
Model Training (TF-IDF + LogReg) + MLflow Tracking
↓
FastAPI Service → POST /predict_issue_type

```

## 📊 Model Performance

**Baseline Model:** TF-IDF (5000 features, bigrams) + Logistic Regression

| Split | Accuracy | Macro F1 |
|-------|----------|----------|
| Train | 97.5%    | 88.1%    |
| Val   | 88.4%    | 31.3%    |
| Test  | 91.4%    | 42.9%    |

*Note: Low macro-F1 on val/test reflects class imbalance (most issues are bugs). Future improvements include oversampling minority classes and fine-tuning transformers.*

---

## 🛠️ Tech Stack

- **Language:** Python 3.11+
- **Data Processing:** pandas, pyarrow
- **ML Framework:** scikit-learn
- **Experiment Tracking:** MLflow
- **API Framework:** FastAPI, uvicorn
- **Data Source:** GitHub REST API
- **Environment:** Anaconda


---

## 📁 Project Structure

```
incident_triage/
├── src/
│   ├── data/
│   │   ├── ingest_github.py       # Fetch issues via GitHub REST API
│   │   └── prepare_issues.py      # Clean and derive labels
│   ├── models/
│   │   └── train_issue_type_mlflow.py  # Train with MLflow tracking
│   └── service/
│       └── app.py                 # FastAPI inference service
├── data/
│   ├── raw/                       # Raw GitHub issues JSON
│   └── processed/                 # Preprocessed parquet files
├── models/
│   └── issue_type_baseline/       # Saved model artifacts
├── mlflow.db                      # MLflow tracking database
├── mlruns/                        # MLflow artifact store
├── requirements.txt
├── .env                           # GitHub token (not committed)
├── .gitignore
└── README.md
```


## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/Denishaji/incident-triage.git
cd incident-triage
```
**Create virtual environment**
```bash
conda create -n incident-triage python=3.11
conda activate incident-triage
```
**Install dependencies**
```bash
pip install -r requirements.txt
```
**Configure GitHub Token**
Create a .env file in project root:
GITHUB_TOKEN=your_github_personal_access_token

Run the Full Pipeline
**Step 1: Ingest raw data**
```bash
python -m src.data.ingest_github
```
**Step 2: Preprocess data**
```bash
python -m src.data.prepare_issues
```
**Step 3: Train model with MLflow tracking**
```bash
python -m src.models.train_issue_type_mlflow
```
**Step 4: View experiments**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```
 Open http://127.0.0.1:5000

**Step 5: Start prediction API**
```bash
uvicorn src.service.app:app --reload
```
Open http://127.0.0.1:8000/docs

---

## 🔬 MLflow Experiment Tracking

View logged experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```
---
##  Add future improvements


## 📈 Future Improvements

-  Add model registry with automated promotion to Staging/Production
-  Fine-tune DistilBERT for improved minority-class performance
-  Implement CI/CD pipeline with GitHub Actions
-  Add monitoring and drift detection
-  Multi-task learning: predict both issue_type and component_team
-  Hyperparameter tuning with Optuna
-  Docker containerization and cloud deployment
---

## 👤 Author

**Deni Shaji**  
MS Data Science, UMass Dartmouth  
[LinkedIn](https://www.linkedin.com/in/deni-shaji-38597635b/) | [GitHub](https://github.com/Denishaji) | [Email](mailto:denishaji308@gmail.com)

*This project demonstrates end-to-end ML engineering skills: data collection, preprocessing, experiment tracking, model versioning, and API deployment.*

---

## 🙏 Acknowledgments

- Data source: [facebook/react](https://github.com/facebook/react) GitHub issues
- Inspired by real-world incident management systems at tech companies

---

