===== FILE: prompts/CHATGPT_HEADER_phase45.md =====
# NHIỆM VỤ: GIAI ĐOẠN 4 + 5 – CreatorFit AI (Streamlit, export, tài liệu)
Phase 1–3 đã nghiệm thu (gate PASS). KHÔNG sửa src/schema.py, preprocessing.py, feature_engineering.py, data_validation.py, models.py, train.py, predict.py, evaluate.py. Dùng đúng hàm có sẵn trong predict.py / export.py (xem code bên dưới); không viết lại logic ML.

## Giai đoạn 4 – `app.py` (đặt ở thư mục gốc; chạy: `streamlit run app.py`)
1. Mặc định tự nạp `data/test.csv` để app chạy được KHÔNG cần upload. Có thêm tùy chọn upload CSV Creator (kiểm tra đủ cột Creator, báo lỗi thân thiện, không crash).
2. Sidebar: selectbox chọn ĐÚNG MỘT `campaign_id`; lấy các cột CAMPAIGN_* của campaign đó từ dữ liệu; creators = các dòng của campaign đã chọn (hoặc creators upload ghép với campaign đã chọn). Không bao giờ trộn nhiều campaign.
3. Chọn mô hình: Ridge / Random Forest / MLP / So sánh cả 3 (dùng `rank_all_models`).
4. Nút "Dự đoán & Xếp hạng" gọi `predict_and_rank(...)`. Hiển thị thông tin campaign, bảng xếp hạng (rank, creator_id, creator_name, fit_score_pred), bar chart top-N bằng matplotlib. Điểm luôn trong 0–100.
5. Nếu cột `target_source` có "synthetic" → hiện banner cảnh báo chứa chữ "SYNTHETIC".
6. `st.download_button` cho CSV và XLSX (dùng export.py; tên file chứa campaign_id + model + timestamp).
7. Tab "Đánh giá mô hình": đọc `outputs/metrics.csv`; thiếu file thì báo, KHÔNG bịa số.
8. `st.cache_resource` cho việc nạp model. Tuyệt đối không gọi `.fit(` hay huấn luyện trong app.
9. Phải chạy được dưới `streamlit.testing.v1.AppTest` mà không exception.

## Giai đoạn 5
1. `requirements.txt`: pandas, numpy, scikit-learn, torch, matplotlib, streamlit, joblib, openpyxl, pytest (ghim phiên bản tối thiểu).
2. `README.md`: mục tiêu; logic một Campaign – nhiều Creator; cấu trúc thư mục; cài đặt (có venv trên macOS: `python3 -m venv .venv`, `source .venv/bin/activate`, `python -m pip install -r requirements.txt`); lệnh theo thứ tự: `python scripts/make_synthetic_data.py --out data` → `python -m src.train` → `python -m src.evaluate` → `streamlit run app.py`; schema dữ liệu; giới hạn đã biết; mục "Kết quả thực vs minh họa" (dữ liệu là synthetic, kết quả chỉ xác nhận pipeline; không dán số liệu viết tay); cách thay bằng dữ liệu thật (chỉ sửa src/schema.py).
3. `tests/test_e2e.py` (pytest, chạy < 60 giây): dùng model/dữ liệu có sẵn (`pytest.skip` nếu thiếu models/); kiểm tra predict_and_rank cho 3 mô hình: điểm trong [0,100], rank 1..n, đúng một campaign, ValueError khi trộn campaign.

## Quy tắc trả lời
- Mỗi file một khối code đầy đủ, không rút gọn. Không giải thích dài.
- Nếu không chạy được lệnh, ghi "chưa chạy", không bịa kết quả.
- Cuối cùng trả lời 1 dòng: app có gọi fit/train ở đâu không?


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


===== FILE: src/predict.py =====
"""Dự đoán + xếp hạng Creator TRONG MỘT Campaign. Chỉ transform, không fit."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models import MLPRegressorTorch
from src.preprocessing import transform
from src.schema import CAMPAIGN_CATEGORICAL, CAMPAIGN_NUMERIC

MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"
_VALID_MODELS: tuple[str, ...] = ("ridge", "rf", "mlp")
_CACHE: dict[str, Any] = {}
_CAMPAIGN_REQ = tuple(CAMPAIGN_NUMERIC) + tuple(CAMPAIGN_CATEGORICAL)


def _clear_cache() -> None:
    _CACHE.clear()


def _get_preprocessor():
    if "preprocessor" not in _CACHE:
        p = MODELS_DIR / "preprocessor.joblib"
        if not p.exists():
            raise FileNotFoundError(f"preprocessor not found: {p}")
        _CACHE["preprocessor"] = joblib.load(p)
    return _CACHE["preprocessor"]


def _get_model(name: str):
    if name not in _VALID_MODELS:
        raise ValueError(f"Invalid model_name {name!r}; expected {_VALID_MODELS}")
    if name in _CACHE:
        return _CACHE[name]
    if name == "ridge":
        _CACHE[name] = joblib.load(MODELS_DIR / "ridge.joblib")
    elif name == "rf":
        _CACHE[name] = joblib.load(MODELS_DIR / "rf.joblib")
    else:
        _CACHE[name] = MLPRegressorTorch.load(MODELS_DIR / "mlp.pt")
    return _CACHE[name]


def predict_and_rank(
    model_name: str,
    campaign: dict | pd.Series,
    creators: pd.DataFrame,
) -> pd.DataFrame:
    if model_name not in _VALID_MODELS:
        raise ValueError(f"model_name must be one of {_VALID_MODELS}, got {model_name!r}")
    if not isinstance(creators, pd.DataFrame):
        raise ValueError("creators must be a pandas.DataFrame")
    if len(creators) == 0:
        raise ValueError("creators is empty")
    if "creator_id" not in creators.columns:
        raise ValueError("creators must contain 'creator_id'")

    camp = campaign.to_dict() if isinstance(campaign, pd.Series) else dict(campaign) if isinstance(campaign, dict) else None
    if camp is None:
        raise ValueError("campaign must be dict or pandas.Series")

    missing = [c for c in _CAMPAIGN_REQ if c not in camp]
    if missing:
        raise ValueError(f"campaign missing required fields: {missing}")

    df = creators.copy()
    uniq = df["campaign_id"].dropna().unique().tolist() if "campaign_id" in df.columns else []
    if len(uniq) > 1:
        raise ValueError(f"creators contains multiple campaign_ids: {uniq}")

    camp_cid = camp.get("campaign_id") or (uniq[0] if uniq else None)
    if camp_cid is None:
        raise ValueError("cannot determine campaign_id")
    if uniq and uniq[0] != camp_cid:
        raise ValueError(f"campaign_id mismatch: {uniq[0]!r} vs {camp_cid!r}")

    df["campaign_id"] = camp_cid
    df["campaign_name"] = camp.get("campaign_name", "")
    for col in _CAMPAIGN_REQ:
        df[col] = camp[col]
    if "creator_name" not in df.columns:
        df["creator_name"] = df["creator_id"].astype(str)

    pre = _get_preprocessor()
    model = _get_model(model_name)
    pred = np.clip(np.asarray(model.predict(transform(pre, df)), dtype=float).ravel(), 0.0, 100.0)

    out = pd.DataFrame({
        "creator_id": df["creator_id"].to_numpy(),
        "creator_name": df["creator_name"].to_numpy(),
        "fit_score_pred": pred,
    }).sort_values("fit_score_pred", ascending=False, kind="mergesort").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1, dtype=int))
    return out


def rank_all_models(campaign, creators) -> dict[str, pd.DataFrame]:
    return {n: predict_and_rank(n, campaign, creators) for n in _VALID_MODELS}

===== FILE: src/export.py =====
"""Xuất kết quả xếp hạng CSV/XLSX."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd


def to_csv(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); df.to_csv(p, index=False); return p


def to_excel(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); df.to_excel(p, index=False); return p


def export_ranking(df: pd.DataFrame, campaign_id: str, model_name: str, out_dir: str | Path = "outputs") -> dict[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"ranking_{campaign_id}_{model_name}_{ts}"
    out = Path(out_dir)
    return {"csv": str(to_csv(df, out / f"{base}.csv")),
            "xlsx": str(to_excel(df, out / f"{base}.xlsx"))}


if __name__ == "__main__":
    demo = pd.DataFrame({"rank": [1, 2], "creator_id": ["C1", "C2"],
                         "creator_name": ["A", "B"], "fit_score_pred": [88.0, 70.0]})
    print(export_ranking(demo, campaign_id="CP_DEMO", model_name="ridge"))

===== FILE: outputs/metrics.csv =====
model,MAE,RMSE,R2,spearman_mean,spearman_std,ndcg@10_mean,p@10_mean,n_campaigns
ridge,5.081765093607593,6.449806239382258,0.9075038170372983,0.9669704405004204,0.007732791807347884,0.9948692548636264,0.8200000000000001,10
rf,3.619727000000001,4.959839742646265,0.9453027158095288,0.9644021805835742,0.010171513133063461,0.996994832728638,0.85,10
mlp,5.182469726816813,6.67251500375756,0.9010058358637997,0.9619787865374683,0.009882768386353694,0.9940664832546506,0.8100000000000002,10
