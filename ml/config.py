from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
ACTIVE_VERSION_FILE = MODELS_DIR / "active_version.txt"

RANDOM_STATE = 42
TEST_SIZE = 0.2
FRAUD_RATE = 0.04

FEATURE_COLUMNS = [
    "amount",
    "hour",
    "is_weekend",
    "txn_type_transfer",
    "txn_type_cash_out",
    "txn_type_debit",
    "balance_delta_origin",
    "balance_delta_dest",
    "velocity_1h",
    "velocity_24h",
    "amount_zscore",
    "merchant_risk_score",
    "device_trust_score",
    "geo_distance_km",
]

TARGET_COLUMN = "is_fraud"
