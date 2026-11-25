import pandas as pd
from src.data_validation import validate_dataframe, check_no_campaign_overlap


def _base_df():
    return pd.DataFrame({
        "creator_id": ["C1"], "creator_name": ["A"],
        "campaign_id": ["CP1"], "campaign_name": ["X"],
        "followers": [1000], "avg_views": [100], "engagement_rate": [0.02],
        "posts_per_week": [5], "avg_video_length_sec": [300], "past_collabs": [3],
        "brand_safety_score": [80], "niche": ["fashion"], "platform": ["instagram"],
        "country": ["VN"], "budget_usd": [500], "min_followers": [500],
        "duration_days": [30], "campaign_niche": ["fashion"],
        "campaign_platform": ["instagram"], "target_country": ["VN"], "goal": ["awareness"],
        "fit_score": [70.0], "target_source": ["synthetic"],
    })


def test_overlap_detected_by_helper():
    assert not check_no_campaign_overlap(_base_df(), _base_df())["ok"]


def test_negative_avg_video_length_not_caught():
    df = _base_df(); df.loc[0, "avg_video_length_sec"] = -10
    r = validate_dataframe(df, "test")
    assert not r["ok"] and any("avg_video_length_sec" in e for e in r["errors"])


def test_negative_past_collabs_not_caught():
    df = _base_df(); df.loc[0, "past_collabs"] = -1
    r = validate_dataframe(df, "test")
    assert not r["ok"] and any("past_collabs" in e for e in r["errors"])


def test_empty_dataframe_should_error():
    r = validate_dataframe(_base_df().iloc[0:0], "empty")
    assert not r["ok"]


def test_duplicate_creator_campaign_caught():
    df = pd.concat([_base_df(), _base_df()], ignore_index=True)
    r = validate_dataframe(df, "dup")
    assert not r["ok"] and any("duplicate" in e.lower() for e in r["errors"])
