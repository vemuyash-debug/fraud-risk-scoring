# Fraud Risk Scoring API

ML fraud-scoring service with **feature pipelines**, **versioned models**, **REST inference**, and a **review dashboard** with **SHAP-based “why flagged”** explanations.

Built to demonstrate production-style ML engineering: training, evaluation (ROC-AUC / PR-AUC), deployment, and analyst-facing explainability.

## Features

- Synthetic PaySim-inspired transaction dataset generator
- Scikit-learn (HistGradientBoosting) training pipeline with versioning (`models/v1`, `v2`, …)
- FastAPI endpoints: `/predict`, `/explain`, `/model/info`, `/drift/check`
- SHAP TreeExplainer for top risk drivers
- Static analyst review UI at `/`
- Docker support

## Quick start

```bash
cd fraud-risk-scoring
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Generate data + train model v1
python scripts/train_all.py

# Run API + dashboard
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** for the review dashboard.  
API docs: **http://localhost:8000/docs**

## API examples

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 48500,
    "hour": 3,
    "is_weekend": 1,
    "txn_type": "transfer",
    "balance_delta_origin": -48500,
    "balance_delta_dest": 48200,
    "velocity_1h": 11,
    "velocity_24h": 38,
    "amount_zscore": 3.4,
    "merchant_risk_score": 0.81,
    "device_trust_score": 0.22,
    "geo_distance_km": 240
  }'
```

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{ ... same payload ... }'
```

## Docker

```bash
docker build -t fraud-risk-scoring .
docker run -p 8000:8000 fraud-risk-scoring
```

## Project structure

```
fraud-risk-scoring/
├── app/                 # FastAPI app + services
├── ml/                  # Training + feature engineering
├── data/                # Synthetic data generator
├── scripts/train_all.py # One-shot train workflow
├── static/              # Review dashboard
├── models/              # Versioned artifacts (generated)
└── Dockerfile
```

## Notes

- Data is **synthetic** for demo/portfolio use; swap in IEEE-CIS Fraud Detection or internal samples for a stronger production narrative.
- Active model version is tracked in `models/active_version.txt`.
