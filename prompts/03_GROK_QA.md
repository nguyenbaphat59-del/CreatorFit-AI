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
