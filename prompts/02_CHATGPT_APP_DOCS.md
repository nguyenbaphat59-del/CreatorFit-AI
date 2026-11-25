# PROMPT CHO CHATGPT – ỨNG DỤNG & TÀI LIỆU (Giai đoạn 4–5)

> Dán `00_CONTRACT.md` ở trên cùng. Chỉ gửi khi Giai đoạn 1–3 đã qua review và `predict.py` chạy được.
> Đính kèm: `src/predict.py`, `src/export.py`, `src/schema.py`, `outputs/metrics.csv`.

---

## GIAI ĐOẠN 4 – Streamlit + Export

Bạn là AI Engineer (Python, Streamlit). Viết `app.py` (và `src/export.py` nếu chưa có).

Yêu cầu giao diện:
1. Sidebar: nạp file Campaign (CSV) hoặc chọn từ `data/test.csv`; chọn ĐÚNG MỘT Campaign (selectbox theo `campaign_id`). Không cho chọn nhiều.
2. Nạp danh sách Creator (CSV upload hoặc mặc định các creator trong test) – validate bằng `validate_dataframe`-tương-đương cho creator, hiển thị lỗi thân thiện.
3. Chọn mô hình: Ridge / Random Forest / MLP (và tùy chọn "So sánh cả 3").
4. Nút "Dự đoán & Xếp hạng" gọi `predict_and_rank(model_name, campaign, creators)`; KHÔNG tự viết lại logic ML trong app.
5. Hiển thị: thông tin Campaign đang chọn, bảng xếp hạng (rank, creator, fit score), top-N bar chart (matplotlib), điểm nằm trong 0–100.
6. Nếu dữ liệu có `target_source == "synthetic"` → hiển thị banner cảnh báo "Dữ liệu/điểm là SYNTHETIC".
7. Nút tải CSV và Excel (xlsx) của bảng xếp hạng; tên file chứa campaign_id + model + timestamp.
8. Tab "Đánh giá mô hình": đọc `outputs/metrics.csv` (nếu chưa có thì thông báo, KHÔNG tự bịa số).
9. Cache model bằng `st.cache_resource`.

Ràng buộc: không trộn nhiều Campaign trong một bảng; không dùng creator_name làm feature.

---

## GIAI ĐOẠN 5 – Tài liệu & đóng gói

1. `requirements.txt`: ghim phiên bản tối thiểu (pandas, numpy, scikit-learn, torch, matplotlib, streamlit, joblib, openpyxl, pytest).
2. `README.md`: mục tiêu; logic một Campaign – nhiều Creator; cấu trúc thư mục; cài đặt; các lệnh chạy theo thứ tự
   (`python scripts/make_synthetic_data.py` → `python src/train.py` → `python src/evaluate.py` → `streamlit run app.py`);
   mô tả schema dữ liệu; giới hạn đã biết; mục "Kết quả thực vs minh họa" nêu rõ số liệu trong README là từ lần chạy nào;
   hướng dẫn thay dữ liệu thật.
3. Một script kiểm tra end-to-end `tests/test_e2e.py` (pytest): sinh dữ liệu nhỏ → train → predict_and_rank → kiểm tra điểm trong [0,100], rank 1..n, một campaign duy nhất.

Cuối cùng liệt kê các lệnh bạn đã chạy thật và kết quả.
