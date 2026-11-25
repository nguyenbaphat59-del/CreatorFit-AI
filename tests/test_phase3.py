"""Tests Phase 3: predict + ranking_metrics + export."""
from __future__ import annotations

from pathlib import Path

import joblib, numpy as np, pandas as pd, pytest, torch

from src import predict as predict_module
from src.evaluate import _ndcg_at_k, _precision_at_k, ranking_metrics, regression_metrics
from src.export import export_ranking
from src.models import MLPRegressorTorch, get_random_forest, get_ridge
from src.preprocessing import fit_preprocessor
from src.schema import RANDOM_STATE


def _make_toy_train(n_per_camp: int = 12, n_camps: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    niches, platforms, countries = ["fashion", "tech"], ["instagram", "tiktok"], ["VN", "US"]
    rows = []
    for ci in range(n_camps):
        for ri in range(n_per_camp):
            followers = int(rng.integers(1_000, 50_000))
            rows.append({
                "creator_id": f"CR_{ci}_{ri}", "creator_name": f"C {ci}-{ri}",
                "campaign_id": f"CAMP_{ci}", "campaign_name": f"Camp {ci}",
                "followers": followers, "avg_views": int(followers * rng.uniform(0.05, 0.5)),
                "engagement_rate": float(rng.uniform(0.01, 0.15)),
                "posts_per_week": float(rng.uniform(1, 20)),
                "avg_video_length_sec": float(rng.uniform(30, 900)),
                "past_collabs": int(rng.integers(0, 20)),
                "brand_safety_score": float(rng.uniform(50, 100)),
                "niche": niches[ri % 2], "platform": platforms[ri % 2], "country": countries[ri % 2],
                "budget_usd": float(rng.uniform(500, 5000)),
                "min_followers": int(rng.integers(500, 5000)),
                "duration_days": int(rng.integers(7, 60)),
                "campaign_niche": niches[ci % 2], "campaign_platform": platforms[ci % 2],
                "target_country": countries[ci % 2], "goal": "awareness",
                "fit_score": float(rng.uniform(20, 95)), "target_source": "synthetic",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def trained(tmp_path, monkeypatch):
    train_df = _make_toy_train()
    pre, X, y = fit_preprocessor(train_df)
    ridge = get_ridge(alpha=1.0); ridge.fit(X, y)
    rf = get_random_forest(n_estimators=5, max_depth=3); rf.fit(X, y)
    mlp = MLPRegressorTorch(hidden_dims=(8, 4), max_epochs=3, patience=2, batch_size=16); mlp.fit(X, y)

    mdir = tmp_path / "models"; mdir.mkdir()
    joblib.dump(pre, mdir / "preprocessor.joblib")
    joblib.dump(ridge, mdir / "ridge.joblib")
    joblib.dump(rf, mdir / "rf.joblib")
    torch.save(mlp.state_dict_with_meta(), mdir / "mlp.pt")

    monkeypatch.setattr(predict_module, "MODELS_DIR", mdir)
    predict_module._clear_cache()
    return train_df


def _campaign(train_df, cid):
    r = train_df[train_df["campaign_id"] == cid].iloc[0]
    return {"campaign_id": r["campaign_id"], "campaign_name": r["campaign_name"],
            "budget_usd": float(r["budget_usd"]), "min_followers": int(r["min_followers"]),
            "duration_days": int(r["duration_days"]),
            "campaign_niche": r["campaign_niche"], "campaign_platform": r["campaign_platform"],
            "target_country": r["target_country"], "goal": r["goal"]}


def _creators(train_df, cid, n=6):
    drop = ["campaign_id", "campaign_name", "fit_score", "target_source",
            "budget_usd", "min_followers", "duration_days",
            "campaign_niche", "campaign_platform", "target_country", "goal"]
    return train_df[train_df["campaign_id"] == cid].head(n).drop(columns=drop).reset_index(drop=True)


# --------- predict ----------
@pytest.mark.parametrize("m", ["ridge", "rf", "mlp"])
def test_predict_and_rank_ok(trained, m):
    camp = _campaign(trained, "CAMP_0"); cr = _creators(trained, "CAMP_0", 6)
    out = predict_module.predict_and_rank(m, camp, cr)
    assert list(out.columns) == ["rank", "creator_id", "creator_name", "fit_score_pred"]
    assert len(out) == 6 and out["rank"].tolist() == [1, 2, 3, 4, 5, 6]
    assert out["fit_score_pred"].between(0, 100).all()
    assert out["fit_score_pred"].is_monotonic_decreasing


def test_predict_reject_multi(trained):
    a, b = "CAMP_0", "CAMP_1"
    part_a = trained[trained["campaign_id"] == a].head(3)
    part_b = trained[trained["campaign_id"] == b].head(3)
    creators = pd.concat([part_a, part_b], ignore_index=True)
    assert creators["campaign_id"].nunique() == 2  # precondition
    with pytest.raises(ValueError, match="multiple campaign_ids"):
        predict_module.predict_and_rank("ridge", _campaign(trained, a), creators)


def test_predict_reject_empty(trained):
    with pytest.raises(ValueError):
        predict_module.predict_and_rank("ridge", _campaign(trained, "CAMP_0"), pd.DataFrame())


def test_predict_reject_bad_model(trained):
    with pytest.raises(ValueError, match="model_name"):
        predict_module.predict_and_rank("xgb", _campaign(trained, "CAMP_0"), _creators(trained, "CAMP_0", 3))


def test_predict_reject_missing_campaign_field(trained):
    c = _campaign(trained, "CAMP_0"); del c["budget_usd"]
    with pytest.raises(ValueError, match="missing required"):
        predict_module.predict_and_rank("ridge", c, _creators(trained, "CAMP_0", 3))


def test_rank_all(trained):
    out = predict_module.rank_all_models(_campaign(trained, "CAMP_0"), _creators(trained, "CAMP_0", 4))
    assert set(out) == {"ridge", "rf", "mlp"}
    for df in out.values():
        assert df["rank"].tolist() == [1, 2, 3, 4]


# --------- ranking_metrics ----------
def test_ndcg_perfect():
    t = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    assert _ndcg_at_k(t, t.copy(), 5) == pytest.approx(1.0)


def test_ndcg_reverse():
    t = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    assert 0.0 <= _ndcg_at_k(t, -t, 5) < 1.0


def test_ndcg_zero_when_true_constant():
    assert _ndcg_at_k(np.zeros(5), np.array([3.0, 1.0, 2.0, 0.0, 4.0]), 5) == 0.0


def test_precision_perfect():
    t = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    assert _precision_at_k(t, t.copy(), 3) == pytest.approx(1.0)


def test_spearman_perfect_and_reverse():
    t = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    df = pd.DataFrame({"campaign_id": ["A"] * 5, "fit_score": t,
                       "pp": t.copy(), "pr": -t})
    mp = ranking_metrics(df, "pp", "fit_score", k=5)
    assert mp["spearman_mean"] == pytest.approx(1.0)
    assert mp["ndcg@k_mean"] == pytest.approx(1.0)
    assert mp["precision@k_mean"] == pytest.approx(1.0)
    mr = ranking_metrics(df, "pr", "fit_score", k=5)
    assert mr["spearman_mean"] == pytest.approx(-1.0)
    assert mr["ndcg@k_mean"] < 1.0


def test_ranking_metrics_multi_campaign_mean():
    t = np.array([10.0, 8.0, 6.0, 4.0])
    rows = []
    for cid, sign in [("X", 1), ("Y", -1)]:
        for ti, pi in zip(t, sign * t):
            rows.append({"campaign_id": cid, "fit_score": ti, "pred": pi})
    m = ranking_metrics(pd.DataFrame(rows), "pred", "fit_score", k=4)
    assert m["spearman_mean"] == pytest.approx(0.0)
    assert m["n_campaigns"] == 2


def test_regression_metrics_perfect():
    y = np.array([10.0, 20.0, 30.0])
    m = regression_metrics(y, y)
    assert m["MAE"] == pytest.approx(0.0) and m["RMSE"] == pytest.approx(0.0) and m["R2"] == pytest.approx(1.0)


# --------- export ----------
def test_export_ranking(tmp_path):
    df = pd.DataFrame({"rank": [1, 2], "creator_id": ["C1", "C2"],
                       "creator_name": ["A", "B"], "fit_score_pred": [88.5, 60.2]})
    paths = export_ranking(df, "CP1", "ridge", out_dir=tmp_path)
    assert Path(paths["csv"]).exists() and Path(paths["xlsx"]).exists()
    assert "CP1" in Path(paths["csv"]).name and "ridge" in Path(paths["csv"]).name