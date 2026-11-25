"""HỢP ĐỒNG DỮ LIỆU (data contract) của CreatorFit AI.

CẢNH BÁO: schema dưới đây là GIẢ ĐỊNH, do chưa có train.csv/test.csv thật và
Technical Handover Document. Khi có dữ liệu thật, chỉ cần sửa file này;
mọi module khác phải import tên cột từ đây, không hard-code.
"""

# --- Định danh (KHÔNG đưa vào model) ---
ID_COLS = ["creator_id", "creator_name", "campaign_id", "campaign_name"]

# --- Feature của Creator ---
CREATOR_NUMERIC = [
    "followers",
    "avg_views",
    "engagement_rate",      # 0-1
    "posts_per_week",
    "avg_video_length_sec",
    "past_collabs",
    "brand_safety_score",   # 0-100
]
CREATOR_CATEGORICAL = ["niche", "platform", "country"]

# --- Feature của Campaign ---
CAMPAIGN_NUMERIC = ["budget_usd", "min_followers", "duration_days"]
CAMPAIGN_CATEGORICAL = ["campaign_niche", "campaign_platform", "target_country", "goal"]

# --- Target ---
TARGET = "fit_score"          # 0-100
TARGET_SOURCE = "target_source"  # "synthetic" | "expert" | "real"; KHÔNG đưa vào model

# --- Cột bắt buộc để validate ---
REQUIRED_COLS = (
    ID_COLS + CREATOR_NUMERIC + CREATOR_CATEGORICAL
    + CAMPAIGN_NUMERIC + CAMPAIGN_CATEGORICAL + [TARGET, TARGET_SOURCE]
)

# Cột không bao giờ được là feature (chống leakage)
FORBIDDEN_FEATURES = set(ID_COLS + [TARGET, TARGET_SOURCE])

RANDOM_STATE = 42
