
# CreatorFit AI

Hệ thống AI hỗ trợ đánh giá mức độ phù hợp giữa Creator và Campaign, sử dụng các mô hình Machine Learning để dự đoán Fit Score và xếp hạng Creator.

## 1. Mục tiêu dự án

CreatorFit AI giải quyết bài toán:

- Một Campaign có nhiều Creator.
- Mỗi dòng dữ liệu đại diện cho một cặp Creator – Campaign.
- Mô hình dự đoán `fit_score` trong khoảng 0–100.
- Xếp hạng Creator theo điểm dự đoán giảm dần trong cùng một Campaign.

Các mô hình được sử dụng:

- Ridge Regression
- Random Forest
- MLP (PyTorch)

Ứng dụng Streamlit cung cấp giao diện lựa chọn Campaign, nạp Creator, dự đoán, xếp hạng và xuất kết quả.

## 2. Nguyên tắc dữ liệu

**Một Campaign – Nhiều Creator**

Không được trộn dữ liệu của nhiều Campaign trong cùng một bảng xếp hạng.

Các trường định danh không được sử dụng làm đặc trưng huấn luyện:

- creator_id
- creator_name
- campaign_id
- campaign_name
- fit_score
- target_source

Điểm dự đoán luôn được giới hạn trong [0, 100].

Dữ liệu tự tạo phải có nhãn `target_source = synthetic`.

## 3. Cấu trúc thư mục

```text
creatorfit/
├── data/
│   ├── train.csv
│   └── test.csv
│
├── scripts/
│   ├── make_synthetic_data.py
│   └── gate.py
│
├── src/
│   ├── __init__.py
│   ├── schema.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── export.py
│
├── models/
│   ├── ridge.joblib
│   ├── rf.joblib
│   ├── mlp.pt
│   └── preprocessor.joblib
│
├── outputs/
│   ├── metrics.csv
│   ├── evaluation_report.md
│   └── rankings.csv / rankings.xlsx
│
├── tests/
│   └── test_e2e.py
│
├── app.py
├── requirements.txt
└── README.md
```

## 4. Công nghệ

- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- PyTorch
- Matplotlib
- Streamlit
- Joblib
- OpenPyXL
- Pytest

## 5. Cài đặt

### macOS

Mở Terminal tại thư mục dự án:

```bash
cd /path/to/creatorfit
```

Tạo môi trường ảo:

```bash
python3 -m venv .venv
```

Kích hoạt môi trường:

```bash
source .venv/bin/activate
```

Cài đặt thư viện:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 6. Chạy dự án theo thứ tự

### Bước 1: Tạo dữ liệu synthetic

```bash
python scripts/make_synthetic_data.py --out data
```

Lệnh tạo dữ liệu huấn luyện và kiểm tra.

### Bước 2: Huấn luyện mô hình

```bash
python -m src.train
```

Quá trình huấn luyện tạo các model và preprocessor trong thư mục `models/`.

### Bước 3: Đánh giá mô hình

```bash
python -m src.evaluate
```

Kết quả đánh giá được lưu trong `outputs/`.

### Bước 4: Chạy ứng dụng

```bash
streamlit run app.py
```

Streamlit sẽ khởi chạy giao diện CreatorFit AI.

## 7. Hướng dẫn sử dụng ứng dụng

### 7.1. Chọn Campaign

- Mở ứng dụng Streamlit.
- Sử dụng Sidebar để chọn một `campaign_id`.
- Thông tin Campaign được lấy từ `data/test.csv`.

Ứng dụng chỉ cho phép lựa chọn một Campaign tại một thời điểm.

### 7.2. Nạp Creator

Có hai phương thức:

1. Sử dụng Creator mặc định trong `data/test.csv`.
2. Upload danh sách Creator bằng file CSV.

File upload phải có các cột Creator bắt buộc theo `src/schema.py`:

- creator_id
- creator_name
- Các trường CREATOR_NUMERIC.
- Các trường CREATOR_CATEGORICAL.

Nếu file có `campaign_id`, giá trị phải khớp Campaign đang chọn.

### 7.3. Chọn mô hình

Người dùng có thể chọn:

- Ridge
- Random Forest
- MLP
- So sánh cả 3

### 7.4. Dự đoán và xếp hạng

Nhấn nút **Dự đoán & Xếp hạng**.

Ứng dụng gọi trực tiếp các hàm trong `src/predict.py`:

- `predict_and_rank(model_name, campaign, creators)`
- `rank_all_models(campaign, creators)`

Kết quả gồm:

- rank
- creator_id
- creator_name
- fit_score_pred

Điểm dự đoán nằm trong khoảng 0–100.

Ứng dụng không huấn luyện mô hình trong quá trình dự đoán.

### 7.5. Xuất kết quả

Sử dụng `src/export.py` để xuất:

- CSV
- Excel XLSX

Tên file có cấu trúc:

```text
ranking_{campaign_id}_{model_name}_{timestamp}.csv
ranking_{campaign_id}_{model_name}_{timestamp}.xlsx
```

### 7.6. Đánh giá mô hình

Tab **Đánh giá mô hình** đọc dữ liệu trực tiếp từ:

```text
outputs/metrics.csv
```

Nếu file không tồn tại, ứng dụng hiển thị thông báo và không tự tạo số liệu.

## 8. Schema dữ liệu

Tên cột được quản lý tập trung trong `src/schema.py`.

### 8.1. Định danh

| Cột | Mô tả |
|---|---|
| creator_id | Mã Creator |
| creator_name | Tên Creator |
| campaign_id | Mã Campaign |
| campaign_name | Tên Campaign |

### 8.2. Creator Numeric

| Cột | Mô tả |
|---|---|
| followers | Số người theo dõi |
| avg_views | Lượt xem trung bình |
| engagement_rate | Tỷ lệ tương tác |
| posts_per_week | Số bài đăng mỗi tuần |
| avg_video_length_sec | Độ dài video trung bình |
| past_collabs | Số lần hợp tác trước đây |
| brand_safety_score | Điểm an toàn thương hiệu |

### 8.3. Creator Categorical

| Cột | Mô tả |
|---|---|
| niche | Lĩnh vực nội dung |
| platform | Nền tảng |
| country | Quốc gia |

### 8.4. Campaign Numeric

| Cột | Mô tả |
|---|---|
| budget_usd | Ngân sách USD |
| min_followers | Số follower tối thiểu |
| duration_days | Thời gian chiến dịch |

### 8.5. Campaign Categorical

| Cột | Mô tả |
|---|---|
| campaign_niche | Lĩnh vực chiến dịch |
| campaign_platform | Nền tảng chiến dịch |
| target_country | Quốc gia mục tiêu |
| goal | Mục tiêu chiến dịch |

### 8.6. Target

| Cột | Mô tả |
|---|---|
| fit_score | Điểm phù hợp 0–100 |
| target_source | Nguồn tạo nhãn |

Các giá trị `target_source` có thể gồm synthetic, expert hoặc real theo hợp đồng dữ liệu.

## 9. Kiểm thử

Chạy kiểm thử:

```bash
pytest -q
```

Kiểm thử end-to-end:

```bash
pytest -q tests/test_e2e.py
```

Kiểm tra gate theo giai đoạn:

```bash
python scripts/gate.py 4
python scripts/gate.py 5
```

Các bài kiểm thử end-to-end kiểm tra:

- Ba mô hình Ridge, RF và MLP.
- Điểm dự đoán trong [0, 100].
- Rank liên tục từ 1 đến n.
- Một Campaign trong một lần xếp hạng.
- Từ chối dữ liệu chứa nhiều Campaign.

Nếu thiếu model đã huấn luyện, bài kiểm thử end-to-end sẽ được skip thay vì tự huấn luyện.

## 10. Giới hạn đã biết

- Kết quả phụ thuộc vào chất lượng dữ liệu đầu vào.
- Dữ liệu synthetic không đại diện hoàn toàn cho thị trường Creator thực tế.
- Fit Score là dự đoán của mô hình, không phải cam kết hiệu quả chiến dịch.
- Mô hình không tự cập nhật khi có dữ liệu mới.
- File Creator upload cần tuân thủ schema.
- Kết quả dự đoán cần được kiểm tra trước khi sử dụng cho quyết định thương mại.

## 11. Kết quả thực vs minh họa

Dữ liệu hiện tại được cung cấp trong dự án có nguồn synthetic.

Các chỉ số trong `outputs/metrics.csv` là kết quả được lưu từ lần đánh giá của bộ dữ liệu và mô hình hiện có. Chúng không được xem là kết quả kiểm chứng trên dữ liệu Creator thực tế.

Không sử dụng số liệu minh họa như kết quả thực nghiệm.

Để xác nhận kết quả mới, cần chạy lại quá trình tạo dữ liệu, huấn luyện và đánh giá theo đúng quy trình.

## 12. Thay thế bằng dữ liệu thật

Khi chuyển sang dữ liệu thực tế:

1. Chuẩn bị dữ liệu Creator và Campaign.
2. Kiểm tra cấu trúc và chất lượng dữ liệu.
3. Đối chiếu tên cột với schema.
4. Chỉ điều chỉnh tên cột và định nghĩa dữ liệu tại `src/schema.py` khi cần thiết.
5. Đảm bảo dữ liệu huấn luyện có target đáng tin cậy.
6. Huấn luyện và đánh giá lại mô hình.
7. Kiểm tra kết quả trên dữ liệu chưa từng dùng để huấn luyện.

Không đưa các trường định danh hoặc target vào feature.

## 13. Nguyên tắc triển khai

- Không trộn nhiều Campaign.
- Không sử dụng creator_name làm feature.
- Preprocessing chỉ fit trên tập train.
- Test chỉ được transform.
- Không huấn luyện trong ứng dụng Streamlit.
- Không tự bịa số liệu đánh giá.
- Mọi điểm dự đoán phải nằm trong [0, 100].

---

**CreatorFit AI – One Campaign, Multiple Creators.**
