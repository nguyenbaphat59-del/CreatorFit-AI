# NHIỆM VỤ: GIAI ĐOẠN 3 – CreatorFit AI (đánh giá + dự đoán + export)
Phase 1–2 đã nghiệm thu (gate PASS). KHÔNG sửa schema.py, preprocessing.py, feature_engineering.py, data_validation.py, models.py, train.py.
Mô hình đã lưu trong models/: ridge.joblib, rf.joblib, mlp.pt (nạp bằng MLPRegressorTorch.load), preprocessor.joblib.

Viết 3 file, đúng chữ ký trong hợp đồng:

## 1. src/predict.py
- `predict_and_rank(model_name, campaign, creators)`; model_name ∈ {"ridge","rf","mlp"}.
- Cache nạp model/preprocessor (không fit lại). Ghép campaign vào từng creator (các cột CAMPAIGN_* từ campaign), dùng `transform` từ preprocessing.py, dự đoán, clip [0,100].
- Trả DataFrame cột: rank (1..n), creator_id, creator_name, fit_score_pred; sắp giảm dần.
- `ValueError` nếu `creators` chứa nhiều campaign_id khác nhau hoặc campaign không hợp lệ; `ValueError` nếu model_name sai hoặc creators rỗng.
- Hàm phụ `rank_all_models(campaign, creators)` trả dict 3 DataFrame.

## 2. src/evaluate.py (chạy bằng `python -m src.evaluate`)
- Nạp data/test.csv, validate, KHÔNG fit gì. Dùng preprocessor + 3 model đã lưu.
- `regression_metrics`: MAE, RMSE, R2. `ranking_metrics(df, score_col, true_col, k=10)`: Spearman, NDCG@k, Precision@k tính TRONG TỪNG campaign rồi lấy mean và std (campaign < k creator thì k = số creator; Spearman hằng số -> bỏ qua campaign đó, không NaN).
- Ghi `outputs/metrics.csv` (mỗi mô hình 1 dòng: model, MAE, RMSE, R2, spearman_mean, spearman_std, ndcg@10_mean, p@10_mean, n_campaigns) và `outputs/evaluation_report.md` (bảng so sánh, nguồn target = giá trị `target_source` đọc từ test.csv; nếu có "synthetic" PHẢI có dòng: "Kết quả trên dữ liệu synthetic chỉ xác nhận pipeline, không phản ánh hiệu năng thực."). Báo cáo ghi mô hình tốt nhất theo số liệu chạy ra, không thiên vị MLP.
- Biểu đồ matplotlib lưu `outputs/pred_vs_true.png`, `outputs/metric_compare.png`.
- Mọi con số lấy từ lần chạy, không viết tay.

## 3. src/export.py
- `to_csv(df, path)`, `to_excel(df, path)` (openpyxl), tự tạo thư mục; có hàm `export_ranking(df, campaign_id, model_name, out_dir="outputs")` trả về đường dẫn csv và xlsx (tên chứa campaign_id, model, timestamp).

## Tests: tests/test_phase3.py (pytest)
- predict: điểm trong [0,100], rank 1..n, giảm dần, từ chối nhiều campaign, creators rỗng, model_name sai.
- ranking_metrics: xếp hạng hoàn hảo cho NDCG=1 và Spearman=1; xếp hạng ngược cho Spearman=-1.
- Test dùng model đã lưu hoặc train nhanh trên dữ liệu nhỏ trong tmp_path.

Cuối cùng: chạy `python -m src.evaluate` và `python -m pytest -q`, dán nguyên log. Trả lời: có chỗ nào fit/transform trên test không?


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


===== FILE: src/models.py =====
"""Ba mô hình: Ridge, RandomForest, MLP PyTorch (CPU-friendly)."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from src.schema import RANDOM_STATE

# =========================================================
# Ridge
# =========================================================
def get_ridge(alpha: float = 1.0) -> Ridge:
    """Trả về Ridge đã cấu hình. Ridge KHÔNG có random_state (deterministic)."""
    return Ridge(alpha=float(alpha))


def tune_ridge(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: Iterable[float] = (0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0),
    n_splits: int = 5,
) -> tuple[float, dict[float, float]]:
    """Chọn alpha bằng GroupKFold theo `campaign_id`, CHỈ trên TRAIN.

    Trả về (best_alpha, {alpha: mean_rmse}).
    Raise ValueError nếu số campaign < 2.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float).ravel()
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError(
            f"tune_ridge: cần ≥ 2 campaign_id để GroupKFold, hiện có {len(unique_groups)}"
        )
    k = min(n_splits, len(unique_groups))
    gkf = GroupKFold(n_splits=k)

    scores: dict[float, float] = {}
    best_alpha, best_rmse = None, float("inf")

    for a in alphas:
        fold_rmse: list[float] = []
        for tr_idx, va_idx in gkf.split(X, y, groups):
            model = get_ridge(alpha=a)
            model.fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[va_idx])
            fold_rmse.append(float(np.sqrt(mean_squared_error(y[va_idx], pred))))
        mean_rmse = float(np.mean(fold_rmse))
        scores[float(a)] = mean_rmse
        if mean_rmse < best_rmse:
            best_rmse, best_alpha = mean_rmse, float(a)

    return best_alpha, scores


# =========================================================
# RandomForest
# =========================================================
def get_random_forest(
    n_estimators: int = 200,
    max_depth: Optional[int] = 20,
    min_samples_leaf: int = 1,
    random_state: int = RANDOM_STATE,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        min_samples_leaf=int(min_samples_leaf),
        random_state=random_state,
        n_jobs=-1,
    )


def tune_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 3,
) -> tuple[dict, dict]:
    """Grid nhỏ + GroupKFold (theo campaign_id), CHỈ trên TRAIN.

    Grid:
      n_estimators ∈ {100, 200}
      max_depth     ∈ {10, None}
      min_samples_leaf ∈ {1, 5}
    Trả về (best_params, {'best_score': rmse, 'all_scores': [...]}).
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float).ravel()
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        # fallback: không tune được GroupKFold, dùng mặc định
        return (
            {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 1},
            {"best_score": float("nan"), "all_scores": []},
        )
    k = min(n_splits, len(unique_groups))
    gkf = GroupKFold(n_splits=k)

    grid = [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": m}
        for n in (100, 200)
        for d in (10, None)
        for m in (1, 5)
    ]

    all_scores: list[dict] = []
    best_params, best_rmse = None, float("inf")

    for params in grid:
        fold_rmse: list[float] = []
        for tr_idx, va_idx in gkf.split(X, y, groups):
            model = get_random_forest(**params)
            model.fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[va_idx])
            fold_rmse.append(float(np.sqrt(mean_squared_error(y[va_idx], pred))))
        mean_rmse = float(np.mean(fold_rmse))
        all_scores.append({"params": params, "rmse": mean_rmse})
        if mean_rmse < best_rmse:
            best_rmse, best_params = mean_rmse, params

    return best_params, {"best_score": best_rmse, "all_scores": all_scores}


# =========================================================
# MLP (PyTorch, CPU)
# =========================================================
class MLPRegressorTorch:
    """MLP 2–3 lớp ẩn, ReLU, Dropout, AdamW, MSE, early stopping.

    - `fit(X, y, groups=...)`: nếu có `groups` (campaign_id) thì tách validation
      THEO campaign_id; nếu không, tách ngẫu nhiên (không khuyến nghị).
    - `predict(X) -> np.ndarray`
    - `save(path)` / `MLPRegressorTorch.load(path)`
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        patience: int = 15,
        batch_size: int = 64,
        val_fraction: float = 0.2,
        random_state: int = RANDOM_STATE,
        device: str = "cpu",
    ) -> None:
        if not (2 <= len(hidden_dims) <= 3):
            raise ValueError("hidden_dims phải có 2–3 lớp ẩn.")
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.val_fraction = float(val_fraction)
        self.random_state = int(random_state)
        self.device = torch.device(device)

        self.model_: Optional[nn.Module] = None
        self.input_dim_: Optional[int] = None
        self.n_epochs_: int = 0
        self.best_val_loss_: float = float("inf")

    # ---- build ----
    def _build(self, in_dim: int) -> nn.Module:
        layers: list[nn.Module] = []
        prev = in_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers).to(self.device)

    # ---- split val theo campaign_id ----
    @staticmethod
    def _split_by_campaign(
        groups: np.ndarray, val_fraction: float, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        unique = np.array(sorted(np.unique(groups)))
        rng = np.random.default_rng(seed)
        rng.shuffle(unique)
        n_val = max(1, int(round(len(unique) * val_fraction)))
        n_val = min(n_val, len(unique) - 1)  # đảm bảo còn ≥1 campaign train
        val_groups = set(unique[:n_val].tolist())
        val_mask = np.array([g in val_groups for g in groups])
        return np.where(~val_mask)[0], np.where(val_mask)[0]

    # ---- fit ----
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "MLPRegressorTorch":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        self.input_dim_ = X.shape[1]
        self.model_ = self._build(self.input_dim_)

        # tách validation
        if X_val is not None and y_val is not None:
            X_tr, y_tr = X, y
            X_va = np.asarray(X_val, dtype=np.float32)
            y_va = np.asarray(y_val, dtype=np.float32).ravel()
        else:
            if groups is not None:
                tr_idx, va_idx = self._split_by_campaign(
                    np.asarray(groups), self.val_fraction, self.random_state
                )
            else:
                n = len(X)
                rng = np.random.default_rng(self.random_state)
                idx = rng.permutation(n)
                n_val = max(1, int(round(n * self.val_fraction)))
                va_idx, tr_idx = idx[:n_val], idx[n_val:]
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]

        X_tr_t = torch.from_numpy(X_tr).to(self.device)
        y_tr_t = torch.from_numpy(y_tr).to(self.device).unsqueeze(1)
        X_va_t = torch.from_numpy(X_va).to(self.device)
        y_va_t = torch.from_numpy(y_va).to(self.device).unsqueeze(1)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        n_train = X_tr_t.shape[0]
        best_state = None
        best_val = float("inf")
        epochs_no_improve = 0
        self.n_epochs_ = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            perm = torch.randperm(n_train, device=self.device)
            for start in range(0, n_train, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb, yb = X_tr_t[idx], y_tr_t[idx]
                optimizer.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()

            # validation
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(X_va_t), y_va_t).item())

            self.n_epochs_ = epoch
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                epochs_no_improve = 0
                best_state = {
                    k: v.detach().clone() for k, v in self.model_.state_dict().items()
                }
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.best_val_loss_ = float(best_val)
        return self

    # ---- predict ----
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("MLPRegressorTorch chưa được fit.")
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            x = torch.from_numpy(X).to(self.device)
            out = self.model_(x).cpu().numpy().ravel()
        return out.astype(np.float64)

    # ---- persist ----
    def state_dict_with_meta(self) -> dict:
        return {
            "config": {
                "hidden_dims": list(self.hidden_dims),
                "dropout": self.dropout,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "batch_size": self.batch_size,
                "val_fraction": self.val_fraction,
                "random_state": self.random_state,
            },
            "input_dim": int(self.input_dim_),
            "state_dict": self.model_.state_dict(),
            "n_epochs": int(self.n_epochs_),
            "best_val_loss": float(self.best_val_loss_),
        }

    @classmethod
    def load(cls, path, device: str = "cpu") -> "MLPRegressorTorch":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        obj = cls(device=device, **ckpt["config"])
        obj.model_ = obj._build(int(ckpt["input_dim"]))
        obj.model_.load_state_dict(ckpt["state_dict"])
        obj.model_.eval()
        obj.input_dim_ = int(ckpt["input_dim"])
        obj.n_epochs_ = int(ckpt.get("n_epochs", 0))
        obj.best_val_loss_ = float(ckpt.get("best_val_loss", float("inf")))
        return obj


if __name__ == "__main__":  # demo nhỏ
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(60, 5))
    y = X[:, 0] * 3.0 + X[:, 1] * -1.5 + rng.normal(scale=0.1, size=60)
    groups = np.repeat(np.arange(6), 10)

    a, scores = tune_ridge(X, y, groups)
    print(f"[demo] best alpha={a} | RMSE={scores[a]:.4f}")

    params, info = tune_random_forest(X, y, groups)
    print(f"[demo] best rf params={params} | RMSE={info['best_score']:.4f}")

    mlp = MLPRegressorTorch(max_epochs=50, patience=5)
    mlp.fit(X, y, groups=groups)
    print(
        f"[demo] mlp epochs={mlp.n_epochs_} | best_val_mse={mlp.best_val_loss_:.4f} "
        f"| pred[:3]={mlp.predict(X[:3]).round(2)}"
    )


===== FILE: src/train.py =====
"""Huấn luyện 3 mô hình. TUYỆT ĐỐI KHÔNG đọc test.csv."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_validation import validate_dataframe
from src.models import (
    MLPRegressorTorch,
    get_random_forest,
    get_ridge,
    tune_random_forest,
    tune_ridge,
)
from src.preprocessing import fit_preprocessor
from src.schema import RANDOM_STATE

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


def _log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CreatorFit AI – train (Phase 2)")
    parser.add_argument("--train", default=str(DATA_DIR / "train.csv"))
    parser.add_argument("--out", default=str(MODELS_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Load + validate TRAIN ----------
    train_df = pd.read_csv(args.train)
    _log(f"loaded train.csv: shape={train_df.shape}")

    vr = validate_dataframe(train_df, name="train.csv")
    _log(f"validate: ok={vr['ok']} n_rows={vr['n_rows']}")
    for w in vr["warnings"]:
        _log(f"warning: {w}")
    for e in vr["errors"]:
        _log(f"ERROR: {e}")
    if not vr["ok"]:
        raise SystemExit("Validation TRAIN thất bại — dừng.")

    groups = train_df["campaign_id"].to_numpy()
    n_campaigns = len(np.unique(groups))
    _log(f"n_campaigns in TRAIN = {n_campaigns}")

    # ---------- 2. Preprocessing fit CHỈ trên TRAIN ----------
    t0 = time.time()
    pre, X_train, y_train = fit_preprocessor(train_df)
    dt_pre = time.time() - t0
    _log(f"preprocessor fitted: X={X_train.shape} y={y_train.shape} | {dt_pre:.2f}s")

    # ---------- 3. Ridge ----------
    _log("tuning Ridge alpha via GroupKFold (campaign_id, TRAIN only) ...")
    t0 = time.time()
    best_alpha, cv_scores = tune_ridge(X_train, y_train, groups)
    dt_tune = time.time() - t0
    _log(
        f"ridge best alpha={best_alpha} "
        f"cv_rmse_train={cv_scores[best_alpha]:.4f} | {dt_tune:.2f}s"
    )
    ridge = get_ridge(alpha=best_alpha)
    t0 = time.time()
    ridge.fit(X_train, y_train)
    _log(f"ridge fit done | {time.time()-t0:.2f}s")
    ridge_path = out_dir / "ridge.joblib"
    joblib.dump(ridge, ridge_path)
    _log(f"saved {ridge_path}")

    # ---------- 4. RandomForest ----------
    _log("tuning RandomForest (grid nhỏ, GroupKFold campaign_id, TRAIN only) ...")
    t0 = time.time()
    rf_params, rf_info = tune_random_forest(X_train, y_train, groups)
    dt_tune_rf = time.time() - t0
    _log(
        f"rf best params={rf_params} "
        f"cv_rmse_train={rf_info['best_score']:.4f} | {dt_tune_rf:.2f}s"
    )
    rf = get_random_forest(**rf_params)
    t0 = time.time()
    rf.fit(X_train, y_train)
    _log(f"rf fit done | {time.time()-t0:.2f}s")
    rf_path = out_dir / "rf.joblib"
    joblib.dump(rf, rf_path)
    _log(f"saved {rf_path}")

    # ---------- 5. MLP ----------
    _log("training MLP (val split by campaign_id, early stopping, CPU) ...")
    t0 = time.time()
    mlp = MLPRegressorTorch(random_state=RANDOM_STATE, device="cpu")
    mlp.fit(X_train, y_train, groups=groups)
    dt_mlp = time.time() - t0
    _log(
        f"mlp fit done | epochs={mlp.n_epochs_} "
        f"best_val_mse={mlp.best_val_loss_:.4f} | {dt_mlp:.2f}s"
    )
    mlp_path = out_dir / "mlp.pt"
    import torch

    torch.save(mlp.state_dict_with_meta(), mlp_path)
    _log(f"saved {mlp_path}")

    # ---------- 6. Preprocessor ----------
    pre_path = out_dir / "preprocessor.joblib"
    joblib.dump(pre, pre_path)
    _log(f"saved {pre_path}")

    _log("DONE.")


if __name__ == "__main__":
    main()
