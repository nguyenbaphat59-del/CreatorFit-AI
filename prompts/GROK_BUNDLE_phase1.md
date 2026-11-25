

===== FILE: prompts/03_GROK_QA.md =====
# PROMPT CHO GROK – QA ĐỘC LẬP (sau mỗi giai đoạn)

> Dán `00_CONTRACT.md` + code của giai đoạn vừa xong + `src/schema.py`. Grok KHÔNG viết lại code, chỉ kiểm tra và viết test.

Bạn là reviewer độc lập (không phải tác giả code). Hãy:

1. **Audit rò rỉ dữ liệu:** liệt kê mọi chỗ có thể fit/learn từ test hoặc từ dữ liệu ngoài train; cột định danh/target lọt vào feature; tách split không theo `campaign_id`.
2. **Đối chiếu hợp đồng:** tên hàm, tham số, kiểu trả về, cột cấm, clip [0,100], một Campaign mỗi bảng xếp hạng.
3. **Tìm lỗi biên:** NaN, chia 0, category lạ, 1 Creator, Creator trùng, Campaign trống, giá trị âm/ngoài khoảng, cột thừa/thiếu/đổi thứ tự.
4. **Viết test pytest tấn công** cho từng lỗi tìm được (mỗi test fail nếu lỗi còn tồn tại).
5. **Kết luận** theo bảng: `Mức (Blocker/Major/Minor) | File:dòng | Vấn đề | Cách sửa`.

Quy tắc:
- Không khen, không tóm tắt lại code. Chỉ nêu vấn đề.
- Không bịa kết quả chạy; nếu không chạy được thì ghi "chưa chạy".
- Nếu không thấy vấn đề ở một mục, ghi "không phát hiện" và nêu đã kiểm tra bằng cách nào.


===== FILE: prompts/00_CONTRACT.md =====
# HỢP ĐỒNG GIỮA CÁC MODULE (dán file này vào ĐẦU mọi prompt gửi DeepSeek/ChatGPT)

## Bối cảnh
CreatorFit AI: MỘT Campaign – NHIỀU Creator. Mỗi dòng dữ liệu = một cặp (Creator, Campaign).
Mô hình dự đoán `fit_score` (0–100). Xếp hạng Creator từ cao xuống thấp TRONG cùng một Campaign.
KHÔNG được thay đổi logic này.

## Ràng buộc cứng
- Không bịa kết quả thực nghiệm; số liệu minh họa phải ghi rõ "ILLUSTRATIVE".
- Target tự tạo phải gắn nhãn synthetic/expert (cột `target_source`).
- Không rò rỉ dữ liệu: fit preprocessing CHỈ trên train, test chỉ transform.
- Không dùng creator_name, creator_id, campaign_id, campaign_name, fit_score, target_source làm feature.
- Không trộn nhiều Campaign trong một bảng xếp hạng.
- Không giả định MLP thắng; báo cáo kết quả đúng như chạy ra.
- Đầu ra dự đoán luôn clip về [0, 100].

## Cấu trúc thư mục (cố định)
```
creatorfit/
├── data/                train.csv, test.csv
├── scripts/             make_synthetic_data.py (đã có)
├── src/
│   ├── schema.py        (đã có – tên cột, KHÔNG hard-code ở nơi khác)
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py        (Ridge, RF, MLP PyTorch)
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── export.py
├── models/              ridge.joblib, rf.joblib, mlp.pt, preprocessor.joblib
├── outputs/             evaluation_report.md, metrics.csv, rankings.csv/xlsx
├── app.py
├── requirements.txt
└── README.md
```

## Chữ ký hàm (KHÔNG đổi tên/tham số nếu chưa báo điều phối)

```python
# data_validation.py
def validate_dataframe(df: pd.DataFrame, name: str) -> dict
    # trả {"ok": bool, "errors": [...], "warnings": [...], "n_rows": int}
    # kiểm tra: cột bắt buộc, kiểu dữ liệu, thiếu giá trị, trùng (creator_id, campaign_id),
    # fit_score trong [0,100], target_source hợp lệ, train/test không chung campaign_id

# feature_engineering.py
def add_features(df: pd.DataFrame) -> pd.DataFrame
    # stateless (không học gì từ dữ liệu), ví dụ: niche_match, platform_match,
    # country_match, follower_ratio = followers/min_followers, log_followers,
    # cost_per_follower = budget_usd/followers, view_rate = avg_views/followers

# preprocessing.py
def build_preprocessor() -> sklearn.compose.ColumnTransformer
    # impute + StandardScaler cho số, impute + OneHotEncoder(handle_unknown="ignore") cho phân loại
def fit_preprocessor(train_df) -> (preprocessor, X_train, y_train)
def transform(preprocessor, df) -> X   # chỉ transform

# models.py
def get_ridge() -> Ridge
def get_random_forest() -> RandomForestRegressor
class MLPRegressorTorch:  # fit(X, y), predict(X) -> np.ndarray, có early stopping bằng validation tách từ TRAIN

# predict.py
def predict_and_rank(model_name: str, campaign: dict | pd.Series,
                     creators: pd.DataFrame) -> pd.DataFrame
    # Ghép campaign vào TỪNG creator -> add_features -> transform -> predict -> clip [0,100]
    # trả DataFrame: rank, creator_id, creator_name, fit_score_pred (sắp giảm dần)
    # ném ValueError nếu campaign không phải đúng 1 campaign

# evaluate.py
def regression_metrics(y_true, y_pred) -> dict      # MAE, RMSE, R2
def ranking_metrics(df, score_col, true_col="fit_score", k=10) -> dict
    # tính TRONG TỪNG campaign rồi lấy trung bình: Spearman, NDCG@k, Precision@k (top-k thật)
# export.py
def to_csv(df, path); def to_excel(df, path)
```

## Nguyên tắc chung
- Python 3.10+, type hints, docstring ngắn, không hằng số "ma thuật" (dùng schema.py).
- Mỗi module chạy/test được độc lập; có `if __name__ == "__main__":` demo nhỏ.
- Seed cố định `RANDOM_STATE = 42`.


===== FILE: src/schema.py =====
"""HỢP ĐỒNG DỮ LIỆU (data contract) của CreatorFit AI.

CẢNH BÁO: schema dưới đây là GIẢ ĐỊNH, do chưa có train.csv/test.csv thật và
Technical Handover Document. Khi có dữ liệu thật, chỉ cần sửa file này;
mọi module khác phải import tên cột từ đây, không hard-code.
"""

# --- Định danh (KHÔNG đưa vào model) ---
ID_COLS = ["creator_id", "creator_name", "campaign_id", "campaign_name"]

# --- Feature của Creator ---
CREATOR_NUMERIC = [
    "followers",
    "avg_views",
    "engagement_rate",      # 0-1
    "posts_per_week",
    "avg_video_length_sec",
    "past_collabs",
    "brand_safety_score",   # 0-100
]
CREATOR_CATEGORICAL = ["niche", "platform", "country"]

# --- Feature của Campaign ---
CAMPAIGN_NUMERIC = ["budget_usd", "min_followers", "duration_days"]
CAMPAIGN_CATEGORICAL = ["campaign_niche", "campaign_platform", "target_country", "goal"]

# --- Target ---
TARGET = "fit_score"          # 0-100
TARGET_SOURCE = "target_source"  # "synthetic" | "expert" | "real"; KHÔNG đưa vào model

# --- Cột bắt buộc để validate ---
REQUIRED_COLS = (
    ID_COLS + CREATOR_NUMERIC + CREATOR_CATEGORICAL
    + CAMPAIGN_NUMERIC + CAMPAIGN_CATEGORICAL + [TARGET, TARGET_SOURCE]
)

# Cột không bao giờ được là feature (chống leakage)
FORBIDDEN_FEATURES = set(ID_COLS + [TARGET, TARGET_SOURCE])

RANDOM_STATE = 42


===== FILE: src/data_validation.py =====
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


===== FILE: src/feature_engineering.py =====
"""Feature engineering — STATELESS (chỉ phép toán theo từng dòng, không học gì)."""
from __future__ import annotations

import numpy as np
import pandas as pd

ENGINEERED_NUMERIC_FEATURES: tuple[str, ...] = (
    "niche_match",
    "platform_match",
    "country_match",
    "follower_ratio",
    "log_followers",
    "cost_per_follower",
    "view_rate",
    "meets_min_followers",
)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Chia an toàn: mẫu = 0 hoặc NaN -> NaN; loại bỏ inf."""
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = num / den
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm các feature phái sinh. Stateless, không mutate input."""
    df = df.copy()

    df["niche_match"]    = (df["niche"]    == df["campaign_niche"]).astype(int)
    df["platform_match"] = (df["platform"] == df["campaign_platform"]).astype(int)
    df["country_match"]  = (df["country"]  == df["target_country"]).astype(int)

    df["follower_ratio"]    = _safe_div(df["followers"],  df["min_followers"])
    df["log_followers"]     = np.log1p(pd.to_numeric(df["followers"], errors="coerce"))
    df["cost_per_follower"] = _safe_div(df["budget_usd"], df["followers"])
    df["view_rate"]         = _safe_div(df["avg_views"],  df["followers"])

    # 1 nếu followers >= min_followers (bao gồm min_followers = 0);
    # NaN nếu một trong hai bị thiếu.
    followers = pd.to_numeric(df["followers"], errors="coerce")
    min_followers = pd.to_numeric(df["min_followers"], errors="coerce")
    meets = (followers >= min_followers).astype(float)
    meets[followers.isna() | min_followers.isna()] = np.nan
    df["meets_min_followers"] = meets

    return df


if __name__ == "__main__":  # demo nhỏ
    demo = pd.DataFrame([{
        "followers": 1000, "avg_views": 100, "budget_usd": 500, "min_followers": 500,
        "niche": "fashion", "campaign_niche": "fashion",
        "platform": "instagram", "campaign_platform": "tiktok",
        "country": "VN", "target_country": "VN",
    }])
    print(add_features(demo))


===== FILE: src/preprocessing.py =====
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


===== FILE: tests/test_phase1.py =====
"""Test phase 1: không rò rỉ, chia cho 0, cột cấm, range validation."""
import numpy as np
import pandas as pd
import pytest

from src.data_validation import validate_dataframe
from src.feature_engineering import add_features
from src.preprocessing import (
    fit_preprocessor, transform,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
from src.schema import FORBIDDEN_FEATURES, TARGET


# ---------- fixtures ----------

def _make_train_df() -> pd.DataFrame:
    """Train: 3 dòng, không NaN, meets_min_followers có biến thiên [1,0,1]."""
    return pd.DataFrame({
        "creator_id":    ["C1", "C2", "C3"],
        "creator_name":  ["A", "B", "C"],
        "campaign_id":   ["CP1", "CP1", "CP1"],
        "campaign_name": ["X", "X", "X"],
        "followers":     [1000, 2000, 5000],
        "avg_views":     [100, 200, 400],
        "engagement_rate": [0.02, 0.03, 0.05],
        "posts_per_week": [5, 3, 7],
        "avg_video_length_sec": [300, 200, 400],
        "past_collabs": [3, 2, 5],
        "brand_safety_score": [80, 90, 85],
        "niche":    ["fashion", "gaming", "food"],
        "platform": ["instagram", "tiktok", "youtube"],
        "country":  ["VN", "ID", "VN"],
        "budget_usd":     [1000, 1000, 1000],
        "min_followers":  [500, 3000, 4000],   # -> meets = 1, 0, 1
        "duration_days":  [30, 30, 30],
        "campaign_niche": ["fashion", "fashion", "fashion"],
        "campaign_platform": ["instagram", "instagram", "instagram"],
        "target_country": ["VN", "VN", "VN"],
        "goal": ["awareness", "awareness", "awareness"],
        "fit_score": [70.0, 50.0, 80.0],
        "target_source": ["synthetic", "synthetic", "synthetic"],
    })


def _make_test_df() -> pd.DataFrame:
    """Test: 2 dòng, không trùng creator/campaign với train, có giá trị lạ."""
    return pd.DataFrame({
        "creator_id":    ["C10", "C11"],
        "creator_name":  ["X", "Y"],
        "campaign_id":   ["CP2", "CP2"],
        "campaign_name": ["Z", "Z"],
        "followers":     [1500, 3000],
        "avg_views":     [150, 300],
        "engagement_rate": [0.025, 0.04],
        "posts_per_week": [4, 6],
        "avg_video_length_sec": [250, 350],
        "past_collabs": [1, 4],
        "brand_safety_score": [85, 88],
        "niche":    ["fashion", "food"],
        "platform": ["instagram", "youtube"],
        "country":  ["VN", "VN"],
        "budget_usd":     [2000, 2000],
        "min_followers":  [1000, 2000],
        "duration_days":  [20, 20],
        "campaign_niche": ["beauty", "beauty"],
        "campaign_platform": ["tiktok", "tiktok"],
        "target_country": ["VN", "VN"],
        "goal": ["engagement", "engagement"],
        "fit_score": [60.0, 70.0],
        "target_source": ["synthetic", "synthetic"],
    })


# ---------- tests cũ (giữ nguyên) ----------

def test_safe_division():
    df = _make_train_df()
    df.loc[df.index[1], "followers"] = 0
    df.loc[df.index[2], "min_followers"] = 0
    out = add_features(df)
    assert pd.isna(out.loc[1, "cost_per_follower"])
    assert pd.isna(out.loc[1, "view_rate"])
    assert pd.isna(out.loc[2, "follower_ratio"])
    for col in ("follower_ratio", "cost_per_follower", "view_rate"):
        vals = out[col].fillna(0).to_numpy()
        assert not np.isinf(vals).any(), f"inf found in {col}"


def test_meets_min_followers():
    """followers >= min_followers -> 1; ngược lại 0; NaN khi thiếu; min=0 -> 1."""
    df = pd.DataFrame({
        "followers":     [1000, 1000, 1000, np.nan, 0],
        "min_followers": [500,  2000, 0,    500,    0],
    })
    # Các cột khác của add_features không cần
    for c in ("avg_views", "budget_usd"):
        df[c] = 0
    for c in ("niche", "campaign_niche", "platform", "campaign_platform",
              "country", "target_country"):
        df[c] = "x"
    out = add_features(df)
    assert out.loc[0, "meets_min_followers"] == 1
    assert out.loc[1, "meets_min_followers"] == 0
    assert out.loc[2, "meets_min_followers"] == 1   # min_followers = 0
    assert pd.isna(out.loc[3, "meets_min_followers"])
    assert out.loc[4, "meets_min_followers"] == 1   # cả hai = 0 -> 1


def test_no_forbidden_features_in_preprocessor():
    all_feats = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    assert all_feats.isdisjoint(FORBIDDEN_FEATURES)
    assert TARGET not in all_feats


def test_unknown_category_does_not_crash():
    train = _make_train_df()
    test = _make_test_df()
    test.loc[test.index[0], "niche"] = "NEVER_SEEN"
    pre, _, _ = fit_preprocessor(train)
    X = transform(pre, test)
    assert X.shape[0] == 2


# ---------- TEST LEAKAGE (viết lại) ----------

def _pandas_expected_scaler(train_feat: pd.DataFrame):
    """Mean/scale độc lập bằng pandas (median-impute + population std)."""
    means, scales = [], []
    for col in NUMERIC_FEATURES:
        v = train_feat[col].astype(float)
        imputed = v.fillna(v.median())
        means.append(imputed.mean())
        s = imputed.std(ddof=0)
        if s == 0:  # sklearn StandardScaler dùng 1.0 cho cột zero-variance
            s = 1.0
        scales.append(s)
    return np.asarray(means), np.asarray(scales)


def test_no_leakage_fit_on_train_only():
    """Train/test tách biệt hoàn toàn. Fit chỉ thấy train.

    Sau khi transform test (giá trị cực lớn + category lạ):
      * scaler.mean_, scaler.scale_ KHÔNG đổi;
      * OHE categories_ KHÔNG đổi;
      * giá trị học được BẰNG tính độc lập từ train bằng pandas.
    """
    train = _make_train_df()
    test = _make_test_df()
    assert set(train["creator_id"]).isdisjoint(test["creator_id"])
    assert set(train["campaign_id"]).isdisjoint(test["campaign_id"])

    pre, Xtr, ytr = fit_preprocessor(train)

    num_pipe = pre.named_transformers_["num"]
    scaler = num_pipe.named_steps["scaler"]
    mean_before = scaler.mean_.copy()
    scale_before = scaler.scale_.copy()

    cat_pipe = pre.named_transformers_["cat"]
    ohe = cat_pipe.named_steps["onehot"]
    cats_before = [c.copy() for c in ohe.categories_]

    # Transform test: giá trị cực lớn + category lạ
    test_mod = test.copy()
    test_mod.loc[test_mod.index[0], "followers"] = 1e12
    test_mod.loc[test_mod.index[0], "engagement_rate"] = 0.999
    test_mod.loc[test_mod.index[0], "niche"] = "NEVER_SEEN_NICHE"
    test_mod.loc[test_mod.index[0], "platform"] = "NEVER_SEEN_PLATFORM"
    _ = transform(pre, test_mod)

    # (a) Tham số KHÔNG đổi
    assert np.allclose(scaler.mean_, mean_before), "scaler.mean_ changed!"
    assert np.allclose(scaler.scale_, scale_before), "scaler.scale_ changed!"
    for cb, ca in zip(cats_before, ohe.categories_):
        assert np.array_equal(cb, ca), "OHE categories_ changed!"

    # (b) Bằng giá trị tính độc lập bằng pandas từ train
    train_feat = add_features(train)
    exp_mean, exp_scale = _pandas_expected_scaler(train_feat)
    assert np.allclose(mean_before, exp_mean, atol=1e-9), (
        f"scaler.mean_ mismatch:\n got={mean_before}\n exp={exp_mean}"
    )
    assert np.allclose(scale_before, exp_scale, atol=1e-9), (
        f"scaler.scale_ mismatch:\n got={scale_before}\n exp={exp_scale}"
    )

    # (c) OHE categories = unique values của train (sorted)
    for j, col in enumerate(CATEGORICAL_FEATURES):
        exp_cats = np.sort(train_feat[col].dropna().astype(str).unique())
        got = np.asarray(cats_before[j], dtype=object)
        assert np.array_equal(np.asarray(got.astype(str)), exp_cats), (
            f"categories_[{j}] for '{col}': got={got}, exp={exp_cats}"
        )


# ---------- TEST VALIDATION RANGE (mỗi điều kiện phải bị bắt) ----------

@pytest.mark.parametrize("col, bad_value", [
    ("engagement_rate", 1.5),
    ("engagement_rate", -0.1),
    ("brand_safety_score", 150.0),
    ("brand_safety_score", -1.0),
    ("followers", -1.0),
    ("avg_views", -1.0),
    ("budget_usd", -1.0),
    ("min_followers", -1.0),
    ("duration_days", -1.0),
    ("posts_per_week", -1.0),
])
def test_validation_catches_range_violations(col, bad_value):
    df = _make_train_df()
    df.loc[df.index[0], col] = bad_value
    result = validate_dataframe(df, name="test")
    assert not result["ok"], f"expected error for {col}={bad_value}"
    assert any(col in e for e in result["errors"]), (
        f"no error mentioned '{col}': {result['errors']}"
    )


def test_validation_passes_clean_data():
    result = validate_dataframe(_make_train_df(), name="train")
    assert result["ok"], result["errors"]


def test_imputation_warning_excludes_ids():
    """NaN ở ID KHÔNG được warn; NaN ở feature THÌ được warn."""
    df = _make_train_df()
    df.loc[df.index[0], "followers"] = np.nan       # feature
    df.loc[df.index[0], "creator_name"] = np.nan    # ID
    df.loc[df.index[0], "campaign_name"] = np.nan   # ID
    result = validate_dataframe(df, name="test")
    warn_text = " | ".join(result["warnings"])
    assert "followers" in warn_text, f"missing warning for followers: {warn_text}"
    assert "creator_name" not in warn_text, f"unexpected warn for ID: {warn_text}"
    assert "campaign_name" not in warn_text, f"unexpected warn for ID: {warn_text}"
