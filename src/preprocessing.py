"""Tiền xử lý: ColumnTransformer (impute + scale + one-hot). Fit CHỈ trên train."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.schema import (
    CREATOR_NUMERIC, CREATOR_CATEGORICAL,
    CAMPAIGN_NUMERIC, CAMPAIGN_CATEGORICAL,
    FORBIDDEN_FEATURES, TARGET,
)
from src.feature_engineering import (
    ENGINEERED_NUMERIC_FEATURES, add_features,
)

NUMERIC_FEATURES: tuple[str, ...] = tuple(
    CREATOR_NUMERIC + CAMPAIGN_NUMERIC + list(ENGINEERED_NUMERIC_FEATURES)
)
CATEGORICAL_FEATURES: tuple[str, ...] = tuple(
    CREATOR_CATEGORICAL + CAMPAIGN_CATEGORICAL
)
ALL_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

_leaked = set(ALL_FEATURES) & FORBIDDEN_FEATURES
if _leaked:
    raise RuntimeError(f"FORBIDDEN features found in feature list: {sorted(_leaked)}")
if TARGET in ALL_FEATURES:
    raise RuntimeError(f"Target '{TARGET}' must not be a feature")


def _make_ohe() -> OneHotEncoder:
    """Tương thích cả sklearn cũ (sparse=) và mới (sparse_output=)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  _make_ohe()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe,     list(NUMERIC_FEATURES)),
            ("cat", categorical_pipe, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _assert_no_forbidden(pre: ColumnTransformer) -> None:
    """Chặn cột cấm trong cả input transformers và output feature names.

    - Bỏ qua mục "remainder" (chứa các cột bị DROP — mặc định chính là cột cấm).
    - Kiểm tra cả ``get_feature_names_out()`` để bắt trường hợp OHE sinh tên
      dạng ``<forbidden>_<value>``.
    """
    if not hasattr(pre, "transformers_"):
        return

    # (a) Kiểm tra input của từng transformer — bỏ "remainder"
    for name, _, cols in pre.transformers_:
        if name == "remainder":
            continue
        if isinstance(cols, (list, tuple)):
            bad = set(cols) & FORBIDDEN_FEATURES
            if bad:
                raise RuntimeError(
                    f"Forbidden columns inside preprocessor '{name}': {sorted(bad)}"
                )

    # (b) Kiểm tra tên feature output
    try:
        feat_names = list(pre.get_feature_names_out())
    except Exception:
        return  # một số transformer cũ không hỗ trợ
    for fn in feat_names:
        if not isinstance(fn, str):
            continue
        if fn in FORBIDDEN_FEATURES:
            raise RuntimeError(f"Forbidden feature name in output: '{fn}'")
        for forb in FORBIDDEN_FEATURES:
            if fn.startswith(forb + "_"):
                raise RuntimeError(
                    f"Output feature '{fn}' derived from forbidden '{forb}'"
                )


def fit_preprocessor(train_df: pd.DataFrame) -> Tuple[ColumnTransformer, "object", "object"]:
    """Fit CHỈ trên train. Trả về (preprocessor, X_train, y_train)."""
    df = add_features(train_df)

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns after add_features: {missing}")

    y = df[TARGET].to_numpy(dtype=float)

    pre = build_preprocessor()
    X = pre.fit_transform(df)

    _assert_no_forbidden(pre)
    return pre, X, y


def transform(pre: ColumnTransformer, df: pd.DataFrame):
    """CHỈ transform — không bao giờ fit trên df."""
    df_feat = add_features(df)
    missing = [c for c in ALL_FEATURES if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Missing feature columns after add_features: {missing}")
    return pre.transform(df_feat)


if __name__ == "__main__":  # demo nhỏ
    demo = pd.DataFrame([{
        "creator_id": "C1", "creator_name": "A",
        "campaign_id": "CP1", "campaign_name": "X",
        "followers": 1000, "avg_views": 100, "engagement_rate": 0.02,
        "posts_per_week": 5, "avg_video_length_sec": 300, "past_collabs": 3,
        "brand_safety_score": 80, "niche": "fashion", "platform": "instagram",
        "country": "VN", "budget_usd": 500, "min_followers": 500, "duration_days": 30,
        "campaign_niche": "fashion", "campaign_platform": "instagram",
        "target_country": "VN", "goal": "awareness",
        "fit_score": 75.0, "target_source": "synthetic",
    }])
    pre, X, y = fit_preprocessor(demo)
    print("X shape:", X.shape, "| y:", y)
