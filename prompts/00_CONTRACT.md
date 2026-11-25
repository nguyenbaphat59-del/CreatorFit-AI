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
