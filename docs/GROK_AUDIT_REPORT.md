# Grok Audit Report — CreatorFit AI

## Metadata
- Date: 2026-11-11
- Reviewer: Grok
- Scope: src/*.py, data/*.csv, outputs/*, contract

## 1. Data leakage audit
RESULT: không phát hiện
- train.py: chỉ read_csv(args.train) — KHÔNG đọc test.csv
- evaluate.py: chỉ transform(pre, test_df) — KHÔNG fit
- predict.py: chỉ transform(pre, df) — KHÔNG fit
- preprocessing.fit_transform chỉ trong fit_preprocessor (gọi từ train)
- models.py .fit chỉ trên train folds (GroupKFold theo campaign_id)
- Grep .fit / .fit_transform / read_csv trong src/*.py: sạch.

## 2. Contract compliance
RESULT: đúng chữ ký chính; clip [0,100]; sort giảm dần; ValueError đúng chỗ.
Minor: contract text nói validate_dataframe kiểm tra train/test overlap, nhưng logic nằm ở check_no_campaign_overlap (tách hàm).

## 3. FORBIDDEN_FEATURES audit
RESULT: không phát hiện lọt.
- schema.FORBIDDEN_FEATURES = ID_COLS + [fit_score, target_source]
- preprocessing: module-level intersect check + _assert_no_forbidden (chặn cả OHE tên dạng forbidden_*)

## 4. Data quality audit
- train.csv: 2400 rows, 40 campaigns (CP0000–CP0039)
- test.csv:  600 rows, 10 campaigns (CP0040–CP0049)
- overlap campaign_id: []
- fit_score: train [7.81, 100], test [19.26, 100] — không NaN
- duplicate (creator_id, campaign_id): 0
- target_source: chỉ "synthetic"
- NaN toàn bộ: 0

## 5. Metrics + Report audit
- metrics.csv khớp evaluation_report.md (làm tròn 4 chữ số).
- Banner SYNTHETIC: có.
- Best RMSE  = rf    (4.9598)
- Best Spearman = ridge (0.9670)
- n_campaigns = 10

## 6. Issues table
| Mức   | File:dòng                          | Vấn đề                                      | Cách sửa |
|-------|------------------------------------|---------------------------------------------|----------|
| Minor | 00_CONTRACT.md vs data_validation  | contract text vs hàm tách overlap           | sửa mô tả contract |
| Minor | predict.py                         | không enforce creator_id dtype string       | thêm check string nếu muốn |
| Minor | evaluate.py:68                     | groupby campaign_id khi có NaN không dropna | df.dropna(subset=["campaign_id"]) |
| Minor | data_validation.py                 | NaN ID không error (đúng EDGE_CASES 31)     | optional strict mode |

**Tổng kết:** Blocker=0, Major=0, Minor=4
