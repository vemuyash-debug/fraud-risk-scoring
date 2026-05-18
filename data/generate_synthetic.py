"""Generate PaySim-inspired synthetic transaction data for fraud detection."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.config import DATA_DIR, FRAUD_RATE, RANDOM_STATE


def generate(n_rows: int = 50_000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    n_fraud = int(n_rows * FRAUD_RATE)
    n_legit = n_rows - n_fraud

    def _block(size: int, fraud: bool) -> pd.DataFrame:
        amount = rng.lognormal(mean=4.5 if not fraud else 6.2, sigma=0.9 if not fraud else 1.4, size=size)
        amount = np.clip(amount, 1, 500_000)

        hour = rng.integers(0, 24, size=size)
        is_weekend = rng.choice([0, 1], size=size, p=[0.72, 0.28])

        txn_type = rng.choice(
            ["payment", "transfer", "cash_out", "debit"],
            size=size,
            p=[0.45, 0.2, 0.2, 0.15] if not fraud else [0.1, 0.45, 0.35, 0.1],
        )

        balance_origin = rng.uniform(500, 80_000, size=size)
        balance_dest = rng.uniform(500, 80_000, size=size)
        balance_delta_origin = -amount + rng.normal(0, 50, size=size)
        balance_delta_dest = amount + rng.normal(0, 50, size=size)

        velocity_1h = rng.poisson(2 if not fraud else 8, size=size)
        velocity_24h = rng.poisson(12 if not fraud else 35, size=size)
        amount_zscore = (amount - amount.mean()) / (amount.std() + 1e-6)

        merchant_risk = rng.beta(2, 8, size=size) if not fraud else rng.beta(6, 3, size=size)
        device_trust = rng.beta(8, 2, size=size) if not fraud else rng.beta(3, 7, size=size)
        geo_distance = rng.exponential(15 if not fraud else 120, size=size)

        return pd.DataFrame(
            {
                "amount": amount,
                "hour": hour,
                "is_weekend": is_weekend,
                "txn_type": txn_type,
                "balance_origin": balance_origin,
                "balance_dest": balance_dest,
                "balance_delta_origin": balance_delta_origin,
                "balance_delta_dest": balance_delta_dest,
                "velocity_1h": velocity_1h,
                "velocity_24h": velocity_24h,
                "amount_zscore": amount_zscore,
                "merchant_risk_score": merchant_risk,
                "device_trust_score": device_trust,
                "geo_distance_km": geo_distance,
                "is_fraud": int(fraud),
            }
        )

    legit = _block(n_legit, fraud=False)
    fraud = _block(n_fraud, fraud=True)
    df = pd.concat([legit, fraud], ignore_index=True)
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50_000)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "transactions.csv"
    generate(args.rows).to_csv(out, index=False)
    print(f"Wrote {out} ({args.rows} rows)")


if __name__ == "__main__":
    main()
