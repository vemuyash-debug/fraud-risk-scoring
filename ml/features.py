import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.config import FEATURE_COLUMNS


def encode_transaction_type(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["txn_type_transfer"] = (out["txn_type"] == "transfer").astype(int)
    out["txn_type_cash_out"] = (out["txn_type"] == "cash_out").astype(int)
    out["txn_type_debit"] = (out["txn_type"] == "debit").astype(int)
    return out


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    encoded = encode_transaction_type(df)
    return encoded[FEATURE_COLUMNS]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [("scale", StandardScaler(), FEATURE_COLUMNS)],
        remainder="drop",
    )


def build_training_pipeline(model) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", model),
        ]
    )
