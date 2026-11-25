# NHIỆM VỤ: GIAI ĐOẠN 2 – CreatorFit AI
Đọc hợp đồng và code Phase 1 (đã được nghiệm thu) bên dưới, rồi thực hiện Giai đoạn 2. Không sửa các file Phase 1 và schema.py.

## GIAI ĐOẠN 2 – Ba mô hình

Viết `src/models.py` và `src/train.py`:
1. Ridge (chọn alpha bằng GroupKFold theo `campaign_id` trên TRAIN, không dùng test).
2. RandomForestRegressor (n_estimators, max_depth, min_samples_leaf: tìm bằng GroupKFold nhỏ gọn hoặc đặt mặc định hợp lý và ghi rõ).
3. MLPRegressorTorch: 2–3 lớp ẩn, ReLU, dropout, AdamW, MSE, early stopping bằng phần validation tách từ TRAIN (tách theo campaign_id), seed cố định, chạy được trên CPU.
4. `train.py`: nạp train.csv, validate, add_features, fit_preprocessor, huấn luyện 3 mô hình, lưu `models/ridge.joblib`, `models/rf.joblib`, `models/mlp.pt`, `models/preprocessor.joblib`.
5. TUYỆT ĐỐI không dùng test.csv trong train.py.
6. In log thời gian huấn luyện; không in ra số liệu "kỳ vọng".


## Cuối mỗi giai đoạn, DeepSeek phải tự trả lời
- Có chỗ nào fit/transform trên test không?
- Có cột định danh/target nào lọt vào feature không?
- Tôi đã chạy thật những lệnh nào, kết quả ra sao (dán nguyên log)?


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

## Quy trình nghiệm thu (tiết kiệm token)
Cuối mỗi giai đoạn N, người dùng chạy: `python scripts/gate.py N` và chỉ dán output (không dán code) cho điều phối viên.
Chỉ dán code lên Claude khi gate FAIL và cần chẩn đoán.


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
