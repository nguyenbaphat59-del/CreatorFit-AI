# App Acceptance Criteria — CreatorFit AI

**Gate Phase C: ≥ 18/20 PASS.**

## A. Khởi động
1. AppTest không exception
2. st.set_page_config
3. st.cache_resource cho models/preprocessor
4. Import src.predict / src.export

## B. Synthetic warning
5. Banner SYNTHETIC
6. Không dùng target_source làm feature

## C. Input validation
7. Upload creators CSV
8. Thiếu cột → st.error thân thiện
9. campaign_id mismatch → lỗi thân thiện
10. Nhiều campaign_id → lỗi rõ
11. File rỗng → không gọi predict
12. Campaign không có creator → không crash

## D. Predict & display
13. Chọn model (ridge/rf/mlp/so sánh 3)
14. So sánh 3 → 3 tab/bảng
15. Cột: rank, creator_id, creator_name, fit_score_pred
16. fit_score_pred ∈ [0,100]
17. Sort giảm dần

## E. Export
18. Download CSV sau khi có kết quả
19. Download Excel
20. Tên file chứa campaign_id + model + timestamp
