import shap

from app.services.model_registry import registry
from ml.config import FEATURE_COLUMNS


def explain_transaction(payload: dict, top_k: int = 6) -> list[dict]:
    X = registry.transaction_to_frame(payload)
    preprocess = registry.pipeline.named_steps["preprocess"]
    model = registry.pipeline.named_steps["model"]
    X_scaled = preprocess.transform(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]

    rows = []
    for feat, val, sv in zip(FEATURE_COLUMNS, X.iloc[0].tolist(), values):
        rows.append({"feature": feat, "value": float(val), "shap_value": float(sv)})

    rows.sort(key=lambda r: abs(r["shap_value"]), reverse=True)
    return rows[:top_k]


def build_summary(top_reasons: list[dict], score: float) -> str:
    if not top_reasons:
        return "No explanation available."
    parts = []
    for r in top_reasons[:3]:
        direction = "increases" if r["shap_value"] > 0 else "decreases"
        parts.append(f"{r['feature']} ({direction} risk)")
    return (
        f"Fraud score {score:.2%}. Primary drivers: "
        + ", ".join(parts)
        + "."
    )
