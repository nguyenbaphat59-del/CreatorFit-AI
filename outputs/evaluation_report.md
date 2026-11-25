# Evaluation Report — CreatorFit AI

## Test set
- Rows: 600
- Campaigns: 10
- target_source: ['synthetic']

> **Kết quả trên dữ liệu synthetic chỉ xác nhận pipeline, không phản ánh hiệu năng thực.**

| model | MAE | RMSE | R2 | spearman_mean | spearman_std | ndcg@10_mean | p@10_mean | n_campaigns |
|---|---|---|---|---|---|---|---|---|
| ridge | 5.0818 | 6.4498 | 0.9075 | 0.9670 | 0.0077 | 0.9949 | 0.8200 | 10 |
| rf | 3.6197 | 4.9598 | 0.9453 | 0.9644 | 0.0102 | 0.9970 | 0.8500 | 10 |
| mlp | 5.1825 | 6.6725 | 0.9010 | 0.9620 | 0.0099 | 0.9941 | 0.8100 | 10 |

## Best model (từ số liệu chạy ra — không thiên vị mô hình)
- RMSE thấp nhất: **rf** (4.9598)
- Spearman@10 cao nhất: **ridge** (0.9670)