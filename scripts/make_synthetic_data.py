"""Sinh dữ liệu SYNTHETIC để kiểm thử pipeline CreatorFit AI.

!!! fit_score ở đây là SYNTHETIC (công thức + nhiễu), KHÔNG phải điểm thật hay
do chuyên gia chấm. Mọi chỉ số tính trên dữ liệu này chỉ chứng minh pipeline
chạy được, KHÔNG phản ánh chất lượng mô hình trên dữ liệu thật.

Train và test dùng các Campaign KHÁC NHAU (tách theo campaign_id) để mô phỏng
tình huống dự đoán cho Campaign chưa từng thấy và tránh rò rỉ.

Chạy: python scripts/make_synthetic_data.py --out data
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import schema as S  # noqa: E402

NICHES = ["beauty", "tech", "food", "travel", "fitness", "gaming", "fashion"]
PLATFORMS = ["tiktok", "youtube", "instagram"]
COUNTRIES = ["VN", "TH", "ID", "PH", "SG"]
GOALS = ["awareness", "conversion", "engagement"]


def make_creators(rng, n):
    followers = np.exp(rng.normal(10.5, 1.2, n)).astype(int).clip(1_000, 20_000_000)
    return pd.DataFrame({
        "creator_id": [f"CR{i:05d}" for i in range(n)],
        "creator_name": [f"Creator {i}" for i in range(n)],
        "followers": followers,
        "avg_views": (followers * rng.uniform(0.05, 0.6, n)).astype(int),
        "engagement_rate": rng.beta(2, 30, n).round(4),
        "posts_per_week": rng.integers(1, 15, n),
        "avg_video_length_sec": rng.integers(10, 600, n),
        "past_collabs": rng.poisson(6, n),
        "brand_safety_score": rng.normal(80, 12, n).clip(20, 100).round(1),
        "niche": rng.choice(NICHES, n),
        "platform": rng.choice(PLATFORMS, n),
        "country": rng.choice(COUNTRIES, n, p=[0.5, 0.15, 0.15, 0.1, 0.1]),
    })


def make_campaigns(rng, n, start=0):
    return pd.DataFrame({
        "campaign_id": [f"CP{i:04d}" for i in range(start, start + n)],
        "campaign_name": [f"Campaign {i}" for i in range(start, start + n)],
        "campaign_niche": rng.choice(NICHES, n),
        "campaign_platform": rng.choice(PLATFORMS, n),
        "target_country": rng.choice(COUNTRIES, n, p=[0.5, 0.15, 0.15, 0.1, 0.1]),
        "goal": rng.choice(GOALS, n),
        "budget_usd": np.exp(rng.normal(8.5, 0.8, n)).astype(int).clip(500, 200_000),
        "min_followers": np.exp(rng.normal(9.5, 1.0, n)).astype(int).clip(1_000, 2_000_000),
        "duration_days": rng.integers(7, 90, n),
    })


def synthetic_score(df, rng):
    s = 20.0
    s += 22 * (df["niche"] == df["campaign_niche"])
    s += 12 * (df["platform"] == df["campaign_platform"])
    s += 8 * (df["country"] == df["target_country"])
    s += 600 * df["engagement_rate"].clip(upper=0.12)
    s += 0.15 * (df["brand_safety_score"] - 70)
    s += 1.0 * df["past_collabs"].clip(upper=15)
    s += np.where(df["followers"] >= df["min_followers"], 6, -10)
    ratio = np.log10(df["followers"] / df["min_followers"]).clip(-1.5, 1.5)
    s += np.where(df["goal"] == "awareness", 3 * ratio, 0)
    cost_pressure = np.log10(df["followers"] / 1000) / np.log10(df["budget_usd"] / 100 + 1)
    s -= 2.0 * cost_pressure.clip(0, 5)
    s += rng.normal(0, 4, len(df))
    return s.clip(0, 100).round(2)


def build(rng, n_campaigns, creators_per_campaign, creator_pool, camp_start):
    camps = make_campaigns(rng, n_campaigns, camp_start)
    parts = []
    for _, c in camps.iterrows():
        sample = creator_pool.sample(creators_per_campaign, random_state=int(rng.integers(1e9)))
        block = sample.assign(**c.to_dict())
        parts.append(block)
    df = pd.concat(parts, ignore_index=True)
    df[S.TARGET] = synthetic_score(df, rng)
    df[S.TARGET_SOURCE] = "synthetic"
    return df[S.REQUIRED_COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--train-campaigns", type=int, default=40)
    ap.add_argument("--test-campaigns", type=int, default=10)
    ap.add_argument("--creators-per-campaign", type=int, default=60)
    ap.add_argument("--seed", type=int, default=S.RANDOM_STATE)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    pool = make_creators(rng, 600)
    train = build(rng, a.train_campaigns, a.creators_per_campaign, pool, 0)
    test = build(rng, a.test_campaigns, a.creators_per_campaign, pool, a.train_campaigns)

    os.makedirs(a.out, exist_ok=True)
    train.to_csv(os.path.join(a.out, "train.csv"), index=False)
    test.to_csv(os.path.join(a.out, "test.csv"), index=False)

    shared = set(train.campaign_id) & set(test.campaign_id)
    assert not shared, f"Campaign trùng giữa train/test: {shared}"
    print(f"train: {train.shape}, test: {test.shape}")
    print(f"campaigns train/test: {train.campaign_id.nunique()}/{test.campaign_id.nunique()} (không trùng)")
    print(f"fit_score range: {train[S.TARGET].min():.1f}-{train[S.TARGET].max():.1f}")
    print("target_source = synthetic (chỉ dùng để test pipeline)")


if __name__ == "__main__":
    main()
