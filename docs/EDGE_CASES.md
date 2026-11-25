# Edge Cases Checklist — CreatorFit AI

> Mỗi mục Grok phải có test case tương ứng trong `tests/test_adversarial.py`.

## A. predict.py (10 case)
1. campaign thiếu `budget_usd` → ValueError "missing required"
2. campaign thiếu `goal` → ValueError "missing required"
3. campaign_id trong creators khác campaign_id truyền vào → ValueError "mismatch"
4. creators có 2 campaign_id → ValueError "multiple campaign_ids"
5. creators là DataFrame rỗng → ValueError "creators is empty"
6. creators không có cột `creator_id` → ValueError
7. creators chỉ có 1 dòng → trả DataFrame 1 dòng, rank=[1]
8. model_name = "xgboost" → ValueError
9. model_name = "Ridge" (viết hoa) → ValueError (case-sensitive)
10. campaign là pd.Series → chấp nhận (to_dict)

## B. evaluate.py (8 case)
11. campaign có 1 creator → Spearman bỏ qua (NaN), n_campaigns không tăng
12. tất cả fit_score giống nhau trong 1 campaign → Spearman bỏ qua
13. k > số creator trong campaign → k = số creator
14. score_col không tồn tại → ValueError
15. true_col không tồn tại → ValueError
16. campaign_id có NaN → groupby bỏ qua
17. test có NaN target → validate_dataframe báo lỗi
18. y_pred có NaN → regression_metrics trả NaN, không crash

## C. preprocessing.py (6 case)
19. category lạ (handle_unknown="ignore") → OHE không crash, vector toàn 0
20. NaN toàn cột numeric → SimpleImputer median = NaN? → cần verify
21. Cột zero-variance → StandardScaler scale_ = 1.0 (sklearn behavior)
22. Cột missing sau add_features → ValueError
23. FORBIDDEN_FEATURES lọt vào input → RuntimeError
24. Tên feature OHE bắt đầu bằng tên cột cấm (vd: `campaign_id_xxx`) → RuntimeError

## D. feature_engineering.py (6 case)
25. `min_followers = 0` → follower_ratio = NaN (safe_div), không inf
26. `followers = 0` → cost_per_follower = NaN, view_rate = NaN
27. `followers` NaN → meets_min_followers = NaN
28. `budget_usd = 0` → cost_per_follower = 0 (không phải inf)
29. Tất cả feature đều 0 → không crash
30. Series có dtype object (số dạng string) → pd.to_numeric(errors="coerce") → NaN

## E. data_validation.py (6 case)
31. NaN ở `creator_id` (ID) → KHÔNG warning impute
32. NaN ở `followers` (feature) → warning "will be imputed"
33. Duplicate (creator_id, campaign_id) → error
34. `fit_score = -1` → error "outside [0,100]"
35. `engagement_rate = 1.5` → error "outside [0,1]"
36. `target_source = "fake"` → error "invalid target_source"

## F. train.py (3 case — không cần test nếu Grok thấy OK)
37. train.csv trùng campaign_id với test.csv → gate 1 phải FAIL
38. train.csv có NaN trong target → validate fail
39. train.csv có <2 campaign_id → tune_ridge raise ValueError

## G. app.py (ChatGPT acceptance — Grok dùng APP_ACCEPTANCE.md)
40. AppTest chạy không exception
41. Upload creators với campaign_id mismatch → báo lỗi thân thiện, không crash
42. Upload creators thiếu cột → báo lỗi thân thiện
43. Download CSV trước khi dự đoán → không hiện nút
44. Chọn "So sánh cả 3" → 3 tab
45. Chọn campaign không có creator → không crash

## Quy ước
- Mức Blocker: crash, sai kết quả, rò rỉ dữ liệu.
- Mức Major: UX kém, thiếu validation, warning thiếu.
- Mức Minor: cosmetic, log message.