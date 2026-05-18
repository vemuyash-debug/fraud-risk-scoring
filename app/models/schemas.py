from typing import Literal

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount")
    hour: int = Field(..., ge=0, le=23)
    is_weekend: int = Field(..., ge=0, le=1)
    txn_type: Literal["payment", "transfer", "cash_out", "debit"]
    balance_delta_origin: float
    balance_delta_dest: float
    velocity_1h: int = Field(..., ge=0)
    velocity_24h: int = Field(..., ge=0)
    amount_zscore: float
    merchant_risk_score: float = Field(..., ge=0, le=1)
    device_trust_score: float = Field(..., ge=0, le=1)
    geo_distance_km: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    fraud_score: float
    risk_level: Literal["low", "medium", "high"]
    is_flagged: bool
    model_version: str
    threshold: float


class ShapContribution(BaseModel):
    feature: str
    value: float
    shap_value: float


class ExplanationResponse(BaseModel):
    fraud_score: float
    risk_level: Literal["low", "medium", "high"]
    is_flagged: bool
    model_version: str
    top_reasons: list[ShapContribution]
    summary: str


class ModelInfoResponse(BaseModel):
    version: str
    trained_at: str
    metrics: dict
    feature_columns: list[str]
