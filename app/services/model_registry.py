import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from ml.config import ACTIVE_VERSION_FILE, MODELS_DIR, ROOT
from ml.features import build_feature_frame, encode_transaction_type


class ModelRegistry:
    def __init__(self) -> None:
        self._pipeline = None
        self._metadata = None
        self._version_dir: Path | None = None

    def _resolve_version_dir(self) -> Path:
        if not ACTIVE_VERSION_FILE.exists():
            raise FileNotFoundError(
                "No trained model found. Run: python scripts/train_all.py"
            )
        version = ACTIVE_VERSION_FILE.read_text().strip()
        version_dir = MODELS_DIR / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Missing model directory: {version_dir}")
        return version_dir

    def load(self) -> None:
        self._version_dir = self._resolve_version_dir()
        self._pipeline = joblib.load(self._version_dir / "pipeline.joblib")
        with open(self._version_dir / "metadata.json") as f:
            self._metadata = json.load(f)

    @property
    def pipeline(self):
        if self._pipeline is None:
            self.load()
        return self._pipeline

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            self.load()
        return self._metadata

    @property
    def version(self) -> str:
        return self.metadata["version"]

    def transaction_to_frame(self, payload: dict) -> pd.DataFrame:
        df = pd.DataFrame([payload])
        encoded = encode_transaction_type(df)
        return build_feature_frame(encoded)

    def predict_proba(self, payload: dict) -> float:
        X = self.transaction_to_frame(payload)
        return float(self.pipeline.predict_proba(X)[0, 1])

    def risk_level(self, score: float) -> str:
        if score >= 0.75:
            return "high"
        if score >= 0.45:
            return "medium"
        return "low"


registry = ModelRegistry()
