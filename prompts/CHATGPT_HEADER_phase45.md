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
