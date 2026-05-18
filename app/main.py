from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.models.schemas import (
    ExplanationResponse,
    ModelInfoResponse,
    PredictionResponse,
    ShapContribution,
    TransactionInput,
)
from app.services.drift import population_drift_score
from app.services.explainer import build_summary, explain_transaction
from app.services.model_registry import registry

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"

app = FastAPI(
    title="Fraud Risk Scoring API",
    description="ML fraud scoring with SHAP explanations for analyst review workflows",
    version="1.0.0",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def load_model() -> None:
    try:
        registry.load()
    except FileNotFoundError:
        # Allow API/docs to start before first training run
        pass


@app.get("/")
def dashboard() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(index)


@app.get("/health")
def health() -> dict:
    ready = registry._pipeline is not None
    return {"status": "ok", "model_loaded": ready}


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    meta = registry.metadata
    return ModelInfoResponse(
        version=meta["version"],
        trained_at=meta["trained_at"],
        metrics=meta["metrics"],
        feature_columns=meta["feature_columns"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: TransactionInput) -> PredictionResponse:
    payload = txn.model_dump()
    score = registry.predict_proba(payload)
    threshold = float(registry.metadata.get("threshold", 0.5))
    return PredictionResponse(
        fraud_score=round(score, 4),
        risk_level=registry.risk_level(score),
        is_flagged=score >= threshold,
        model_version=registry.version,
        threshold=threshold,
    )


@app.post("/explain", response_model=ExplanationResponse)
def explain(txn: TransactionInput) -> ExplanationResponse:
    payload = txn.model_dump()
    score = registry.predict_proba(payload)
    threshold = float(registry.metadata.get("threshold", 0.5))
    reasons = explain_transaction(payload)
    top = [ShapContribution(**r) for r in reasons]
    return ExplanationResponse(
        fraud_score=round(score, 4),
        risk_level=registry.risk_level(score),
        is_flagged=score >= threshold,
        model_version=registry.version,
        top_reasons=top,
        summary=build_summary(reasons, score),
    )


@app.post("/drift/check")
def drift_check(transactions: list[TransactionInput]) -> dict:
    if len(transactions) < 5:
        raise HTTPException(status_code=400, detail="Provide at least 5 transactions")
    batch = [t.model_dump() for t in transactions]
    return population_drift_score(batch)
