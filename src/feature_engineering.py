"""Feature engineering — STATELESS (chỉ phép toán theo từng dòng, không học gì)."""
from __future__ import annotations

import numpy as np
import pandas as pd

ENGINEERED_NUMERIC_FEATURES: tuple[str, ...] = (
    "niche_match",
    "platform_match",
    "country_match",
    "follower_ratio",
    "log_followers",
    "cost_per_follower",
    "view_rate",
    "meets_min_followers",
)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Chia an toàn: mẫu = 0 hoặc NaN -> NaN; loại bỏ inf."""
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = num / den
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Thêm các feature phái sinh. Stateless, không mutate input."""
    df = df.copy()

    df["niche_match"]    = (df["niche"]    == df["campaign_niche"]).astype(int)
    df["platform_match"] = (df["platform"] == df["campaign_platform"]).astype(int)
    df["country_match"]  = (df["country"]  == df["target_country"]).astype(int)

    df["follower_ratio"]    = _safe_div(df["followers"],  df["min_followers"])
    df["log_followers"]     = np.log1p(pd.to_numeric(df["followers"], errors="coerce"))
    df["cost_per_follower"] = _safe_div(df["budget_usd"], df["followers"])
    df["view_rate"]         = _safe_div(df["avg_views"],  df["followers"])

    # 1 nếu followers >= min_followers (bao gồm min_followers = 0);
    # NaN nếu một trong hai bị thiếu.
    followers = pd.to_numeric(df["followers"], errors="coerce")
    min_followers = pd.to_numeric(df["min_followers"], errors="coerce")
    meets = (followers >= min_followers).astype(float)
    meets[followers.isna() | min_followers.isna()] = np.nan
    df["meets_min_followers"] = meets

    return df


if __name__ == "__main__":  # demo nhỏ
    demo = pd.DataFrame([{
        "followers": 1000, "avg_views": 100, "budget_usd": 500, "min_followers": 500,
        "niche": "fashion", "campaign_niche": "fashion",
        "platform": "instagram", "campaign_platform": "tiktok",
        "country": "VN", "target_country": "VN",
    }])
    print(add_features(demo))
