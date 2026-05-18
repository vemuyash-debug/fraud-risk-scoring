#!/usr/bin/env python3
"""Generate synthetic data and train a new model version."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.generate_synthetic import generate
from ml.config import DATA_DIR
from ml.train import next_version_dir, train


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "transactions.csv"
    generate(50_000).to_csv(csv_path, index=False)
    print(f"Generated {csv_path}")

    version_dir = next_version_dir()
    meta = train(csv_path, version_dir)
    print(f"Trained {version_dir.name} | ROC-AUC={meta['metrics']['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
