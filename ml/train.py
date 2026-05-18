"""Train fraud classifier and version artifacts under models/v{N}/."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

from ml.config import (
    ACTIVE_VERSION_FILE,
    DATA_DIR,
    FEATURE_COLUMNS,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from ml.features import build_feature_frame, build_training_pipeline


def next_version_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    existing = [p for p in MODELS_DIR.iterdir() if p.is_dir() and p.name.startswith("v")]
    nums = []
    for p in existing:
        try:
            nums.append(int(p.name[1:]))
        except ValueError:
            continue
    n = max(nums, default=0) + 1
    out = MODELS_DIR / f"v{n}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def train(data_path: Path, version_dir: Path) -> dict:
    df = pd.read_csv(data_path)
    if "txn_type" not in df.columns:
        raise ValueError("Dataset must include txn_type column")

    X = build_feature_frame(df)
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    classifier = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.08,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    pipeline = build_training_pipeline(classifier)
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "classification_report": classification_report(y_test, preds, output_dict=True),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "fraud_rate_train": float(y_train.mean()),
    }

    joblib.dump(pipeline, version_dir / "pipeline.joblib")

    reference_stats = {
        col: {
            "mean": float(X_train[col].mean()),
            "std": float(X_train[col].std() + 1e-9),
        }
        for col in FEATURE_COLUMNS
    }

    metadata = {
        "version": version_dir.name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "reference_stats": reference_stats,
        "threshold": 0.5,
    }

    with open(version_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    ACTIVE_VERSION_FILE.write_text(version_dir.name)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR / "transactions.csv")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Missing {args.data}. Run: python scripts/train_all.py")

    version_dir = next_version_dir()
    meta = train(args.data, version_dir)
    print(f"Saved {version_dir.name}")
    print(f"ROC-AUC: {meta['metrics']['roc_auc']:.4f}")
    print(f"PR-AUC:  {meta['metrics']['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
