"""End-to-end tests: load models có sẵn, chạy predict_and_rank cho 3 models.
Skip nếu thiếu models/ hoặc data/test.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.predict import predict_and_rank, rank_all_models, _clear_cache

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REQUIRED_MODELS = ("ridge.joblib", "rf.joblib", "mlp.pt", "preprocessor.joblib")


@pytest.fixture(scope="module")
def artifacts_ready():
    test_csv = DATA_DIR / "test.csv"
    if not test_csv.exists():
        pytest.skip(f"missing {test_csv}")
    missing = [m for m in REQUIRED_MODELS if not (MODELS_DIR / m).exists()]
    if missing:
        pytest.skip(f"missing models: {missing} (run python -m src.train)")
    _clear_cache()
    return True


@pytest.fixture(scope="module")
def test_df(artifacts_ready):
    return pd.read_csv(DATA_DIR / "test.csv")


def _pick_one_campaign(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    cid = df["campaign_id"].iloc[0]
    sub = df[df["campaign_id"] == cid].copy()
    row = sub.iloc[0]
    campaign = {
        "campaign_id": cid,
        "campaign_name": row["campaign_name"],
        "budget_usd": float(row["budget_usd"]),
        "min_followers": int(row["min_followers"]),
        "duration_days": int(row["duration_days"]),
        "campaign_niche": row["campaign_niche"],
        "campaign_platform": row["campaign_platform"],
        "target_country": row["target_country"],
        "goal": row["goal"],
    }
    drop = ["campaign_id", "campaign_name", "fit_score", "target_source",
            "budget_usd", "min_followers", "duration_days",
            "campaign_niche", "campaign_platform", "target_country", "goal"]
    creators = sub.drop(columns=drop).reset_index(drop=True)
    return campaign, creators


@pytest.mark.parametrize("model_name", ["ridge", "rf", "mlp"])
def test_e2e_predict_and_rank(test_df, model_name):
    campaign, creators = _pick_one_campaign(test_df)
    out = predict_and_rank(model_name, campaign, creators)
    assert list(out.columns) == ["rank", "creator_id", "creator_name", "fit_score_pred"]
    assert len(out) == len(creators)
    assert out["rank"].tolist() == list(range(1, len(out) + 1))
    assert out["fit_score_pred"].between(0, 100).all()
    assert out["fit_score_pred"].is_monotonic_decreasing


def test_e2e_rank_all_models(test_df):
    campaign, creators = _pick_one_campaign(test_df)
    out = rank_all_models(campaign, creators)
    assert set(out.keys()) == {"ridge", "rf", "mlp"}
    for name, df in out.items():
        assert len(df) == len(creators), f"{name}: wrong len"
        assert df["rank"].tolist() == list(range(1, len(df) + 1)), f"{name}: rank wrong"
        assert df["fit_score_pred"].between(0, 100).all(), f"{name}: out of range"
        assert df["fit_score_pred"].is_monotonic_decreasing, f"{name}: not sorted"


def test_e2e_reject_multi_campaign(test_df):
    cids = test_df["campaign_id"].unique()
    if len(cids) < 2:
        pytest.skip("need ≥2 campaigns in test.csv")
    a, b = cids[0], cids[1]
    part_a = test_df[test_df["campaign_id"] == a].head(3)
    part_b = test_df[test_df["campaign_id"] == b].head(3)
    creators = pd.concat([part_a, part_b], ignore_index=True)
    assert creators["campaign_id"].nunique() == 2
    campaign, _ = _pick_one_campaign(test_df[test_df["campaign_id"] == a])
    with pytest.raises(ValueError, match="multiple campaign_ids"):
        predict_and_rank("ridge", campaign, creators)
