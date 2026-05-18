import pandas as pd

from app.services.model_registry import registry
from ml.config import FEATURE_COLUMNS
from ml.features import build_feature_frame, encode_transaction_type


def population_drift_score(batch: list[dict]) -> dict:
    ref = registry.metadata["reference_stats"]
    df = build_feature_frame(encode_transaction_type(pd.DataFrame(batch)))

    drifted = []
    max_z = 0.0
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        mean = float(df[col].mean())
        ref_mean = ref[col]["mean"]
        ref_std = ref[col]["std"]
        z = abs(mean - ref_mean) / ref_std
        max_z = max(max_z, z)
        if z > 2.5:
            drifted.append({"feature": col, "z_score": round(z, 2)})

    status = "stable"
    if max_z > 3.5:
        status = "high_drift"
    elif max_z > 2.5:
        status = "moderate_drift"

    return {
        "status": status,
        "max_z_score": round(max_z, 2),
        "drifted_features": drifted,
    }
