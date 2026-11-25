"""Validation dữ liệu theo data contract. Không raise trừ khi thiếu cột bắt buộc."""
from __future__ import annotations

import pandas as pd

from src.schema import (
    CREATOR_NUMERIC, CAMPAIGN_NUMERIC,
    ID_COLS, TARGET, TARGET_SOURCE, REQUIRED_COLS,
)

VALID_TARGET_SOURCES = {"synthetic", "expert", "real"}

# Khoảng giá trị hợp lệ (bao gồm 2 đầu)
RANGE_CHECKS: dict[str, tuple[float, float]] = {
    "engagement_rate":    (0.0, 1.0),
    "brand_safety_score": (0.0, 100.0),
}

# Cột phải >= 0
NON_NEGATIVE_COLS: tuple[str, ...] = (
    "followers", "avg_views", "budget_usd",
    "min_followers", "duration_days", "posts_per_week",
    "avg_video_length_sec", "past_collabs",
)

# Cột feature = REQUIRED_COLS trừ ID, target, target_source
_FEATURE_COLS: tuple[str, ...] = tuple(
    c for c in REQUIRED_COLS
    if c not in set(ID_COLS) and c not in {TARGET, TARGET_SOURCE}
)


def validate_dataframe(df: pd.DataFrame, name: str) -> dict:
    """Kiểm tra 1 DataFrame theo contract.

    - Raise ValueError CHỈ KHI: không phải DataFrame, hoặc thiếu cột bắt buộc.
    - Mọi lỗi khác được gom vào dict.errors (ok=False nhưng không raise).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name}: expected pandas.DataFrame, got {type(df).__name__}")

    errors: list[str] = []
    warnings: list[str] = []

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")

    n_rows = int(len(df))
    if n_rows == 0:
        errors.append(f"{name}: dataframe is empty")

    # 1) Kiểu dữ liệu số
    numeric_cols = CREATOR_NUMERIC + CAMPAIGN_NUMERIC + [TARGET]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(
                f"{name}: column '{col}' expected numeric, got dtype={df[col].dtype}"
            )

    # 2) fit_score trong [0, 100]
    if pd.api.types.is_numeric_dtype(df[TARGET]):
        s = df[TARGET]
        bad = ((s < 0) | (s > 100)) & s.notna()
        if bad.any():
            errors.append(
                f"{name}: {int(bad.sum())} rows have {TARGET} outside [0,100]"
            )

    # 3) Khoảng giá trị cố định
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = df[col]
        bad = ((s < lo) | (s > hi)) & s.notna()
        if bad.any():
            errors.append(
                f"{name}: {int(bad.sum())} rows have {col} outside [{lo},{hi}]"
            )

    # 4) Cột không âm
    for col in NON_NEGATIVE_COLS:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        s = df[col]
        bad = (s < 0) & s.notna()
        if bad.any():
            errors.append(f"{name}: {int(bad.sum())} rows have {col} < 0")

    # 5) target_source hợp lệ
    ts = df[TARGET_SOURCE]
    bad_src_mask = (~ts.isin(VALID_TARGET_SOURCES)) & ts.notna()
    if bad_src_mask.any():
        bad_vals = sorted(map(str, ts[bad_src_mask].unique().tolist()))
        errors.append(f"{name}: invalid {TARGET_SOURCE} values: {bad_vals}")

    # 6) Giá trị thiếu
    missing_cols = [c for c in REQUIRED_COLS if df[c].isna().any()]
    if missing_cols:
        if TARGET in missing_cols:
            errors.append(f"{name}: missing values in target column '{TARGET}'")
        # Warning imputation: CHỈ cột feature, KHÔNG áp dụng ID / target_source
        feat_missing = [c for c in missing_cols if c in _FEATURE_COLS]
        if feat_missing:
            warnings.append(
                f"{name}: missing values in {feat_missing} (will be imputed)"
            )

    # 7) Trùng (creator_id, campaign_id)
    dup_mask = df.duplicated(subset=["creator_id", "campaign_id"], keep=False)
    if dup_mask.any():
        errors.append(
            f"{name}: {int(dup_mask.sum())} duplicate (creator_id, campaign_id) rows"
        )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "n_rows": n_rows,
    }


def check_no_campaign_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Kiểm tra train/test không chia sẻ campaign_id (chống rò rỉ xuyên split)."""
    train_camps = set(train_df["campaign_id"].dropna().unique())
    test_camps = set(test_df["campaign_id"].dropna().unique())
    overlap = sorted(train_camps & test_camps)
    return {"ok": len(overlap) == 0, "overlap_campaign_ids": overlap}


if __name__ == "__main__":  # demo nhỏ
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        df = pd.read_csv(path)
        print(validate_dataframe(df, name=path))
