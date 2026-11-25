# PROMPT CHO DEEPSEEK – LÕI ML (Giai đoạn 1–3)

> Dán nội dung `00_CONTRACT.md` ở trên cùng, đính kèm `src/schema.py`, `scripts/make_synthetic_data.py`,
> và 10 dòng đầu `data/train.csv`. Gửi LẦN LƯỢT từng giai đoạn, chỉ sang giai đoạn sau khi đã qua review.

---

## GIAI ĐOẠN 1 – Validation, Preprocessing, Feature Engineering

Bạn là AI Engineer (Python, Pandas, scikit-learn). Hãy viết 3 file theo đúng hợp đồng:
`src/data_validation.py`, `src/feature_engineering.py`, `src/preprocessing.py`.

Yêu cầu:
1. `validate_dataframe` kiểm tra đầy đủ các mục trong hợp đồng, trả về dict, không `raise` trừ khi thiếu cột bắt buộc.
2. `add_features` stateless; chỉ dùng phép tính theo từng dòng; xử lý chia cho 0 và NaN.
3. `build_preprocessor` dùng ColumnTransformer, danh sách cột lấy từ `schema.py` + cột mới của `add_features`.
4. `fit_preprocessor` chỉ fit trên train; `transform` không bao giờ fit.
5. Có assert/kiểm tra không cột nào thuộc `FORBIDDEN_FEATURES` lọt vào X.
6. Kèm `tests/test_phase1.py` (pytest): test không rò rỉ (fit trên train, thử bằng cách đổi test không làm đổi preprocessor), test chia cho 0, test cột cấm.

Đầu ra: code đầy đủ từng file trong khối code riêng + giải thích ngắn 5 dòng về quyết định tiền xử lý.

---

## GIAI ĐOẠN 2 – Ba mô hình

Viết `src/models.py` và `src/train.py`:
1. Ridge (chọn alpha bằng GroupKFold theo `campaign_id` trên TRAIN, không dùng test).
2. RandomForestRegressor (n_estimators, max_depth, min_samples_leaf: tìm bằng GroupKFold nhỏ gọn hoặc đặt mặc định hợp lý và ghi rõ).
3. MLPRegressorTorch: 2–3 lớp ẩn, ReLU, dropout, AdamW, MSE, early stopping bằng phần validation tách từ TRAIN (tách theo campaign_id), seed cố định, chạy được trên CPU.
4. `train.py`: nạp train.csv, validate, add_features, fit_preprocessor, huấn luyện 3 mô hình, lưu `models/ridge.joblib`, `models/rf.joblib`, `models/mlp.pt`, `models/preprocessor.joblib`.
5. TUYỆT ĐỐI không dùng test.csv trong train.py.
6. In log thời gian huấn luyện; không in ra số liệu "kỳ vọng".

---

## GIAI ĐOẠN 3 – Đánh giá

Viết `src/evaluate.py`:
1. Nạp test.csv, validate, add_features, `transform` bằng preprocessor đã lưu.
2. Metric hồi quy: MAE, RMSE, R² cho cả 3 mô hình.
3. Metric xếp hạng: tính TRONG TỪNG campaign (Spearman, NDCG@10, Precision@10), rồi trung bình + độ lệch chuẩn qua các campaign.
4. Ghi `outputs/metrics.csv` và `outputs/evaluation_report.md` (bảng so sánh, biểu đồ matplotlib scatter dự đoán vs thật, bar so sánh metric).
5. Báo cáo PHẢI ghi nguồn target (`target_source`) và nếu là synthetic phải có dòng cảnh báo: "Kết quả trên dữ liệu synthetic chỉ xác nhận pipeline, không phản ánh hiệu năng thực."
6. Không viết tay bất kỳ con số nào; mọi số lấy từ lần chạy.

---

## Cuối mỗi giai đoạn, DeepSeek phải tự trả lời
- Có chỗ nào fit/transform trên test không?
- Có cột định danh/target nào lọt vào feature không?
- Tôi đã chạy thật những lệnh nào, kết quả ra sao (dán nguyên log)?
