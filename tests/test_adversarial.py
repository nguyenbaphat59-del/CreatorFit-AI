"""Adversarial / edge-case tests — CreatorFit AI.

Checklist nguồn: docs/EDGE_CASES.md (A–E).
Không load models thật nếu không cần; dùng fixture nhỏ trong tmp_path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_validation import validate_dataframe, check_no_campaign_overlap
from src.feature_engineering import add_features, _safe_div
from src.evaluate import ranking_metrics, regression_metrics
from src.schema import (
    REQUIRED_COLS, FORBIDDEN_FEATURES, TARGET, TARGET_SOURCE,
    CREATOR_NUMERIC, CAMPAIGN_NUMERIC, CREATOR_CATEGORICAL, CAMPAIGN_CATEGORICAL,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _base_row(**overrides) -> dict:
    row = {
        "creator_id": "C1",
        "creator_name": "Creator One",
        "campaign_id": "CP1",
        "campaign_name": "Camp One",
        "followers": 10_000,
        "avg_views": 1_000,
        "engagement_rate": 0.05,
        "posts_per_week": 5.0,
        "avg_video_length_sec": 300.0,
        "past_collabs": 3,
        "brand_safety_score": 80.0,
        "niche": "fashion",
        "platform": "instagram",
        "country": "VN",
        "budget_usd": 5_000.0,
        "min_followers": 1_000,
        "duration_days": 30,
        "campaign_niche": "fashion",
        "campaign_platform": "instagram",
        "target_country": "VN",
        "goal": "awareness",
        "fit_score": 70.0,
        "target_source": "synthetic",
    }
    row.update(overrides)
    return row


def _make_df(n: int = 3, **overrides) -> pd.DataFrame:
    rows = []
    for i in range(n):
        kwargs = {
            "creator_id": f"C{i}",
            "creator_name": f"Creator {i}",
            "fit_score": 50.0 + i * 10,
        }
        kwargs.update(overrides)
        r = _base_row(**kwargs)
        rows.append(r)
    return pd.DataFrame(rows)


# ===========================================================================
# A. predict.py (10 cases) — logic validation without real model files
# ===========================================================================

class TestPredictAdversarial:
    """Test input-validation branch of predict_and_rank (no model load)."""

    def _import_predict(self):
        from src import predict as pred
        return pred

    def test_campaign_missing_budget_usd(self):
        pred = self._import_predict()
        camp = _base_row()
        del camp["budget_usd"]
        creators = _make_df(2)
        with pytest.raises(ValueError, match="missing required"):
            pred.predict_and_rank("ridge", camp, creators)

    def test_campaign_missing_goal(self):
        pred = self._import_predict()
        camp = _base_row()
        del camp["goal"]
        creators = _make_df(2)
        with pytest.raises(ValueError, match="missing required"):
            pred.predict_and_rank("ridge", camp, creators)

    def test_campaign_id_mismatch(self):
        pred = self._import_predict()
        camp = _base_row(campaign_id="CP_OTHER")
        creators = _make_df(2, campaign_id="CP1")
        with pytest.raises(ValueError, match="mismatch"):
            pred.predict_and_rank("ridge", camp, creators)

    def test_multiple_campaign_ids_in_creators(self):
        pred = self._import_predict()
        camp = _base_row(campaign_id="CP1")
        a = _make_df(2, campaign_id="CP1")
        b = _make_df(2, campaign_id="CP2")
        creators = pd.concat([a, b], ignore_index=True)
        with pytest.raises(ValueError, match="multiple campaign_ids"):
            pred.predict_and_rank("ridge", camp, creators)

    def test_creators_empty(self):
        pred = self._import_predict()
        camp = _base_row()
        with pytest.raises(ValueError, match="empty"):
            pred.predict_and_rank("ridge", camp, pd.DataFrame())

    def test_creators_missing_creator_id(self):
        pred = self._import_predict()
        camp = _base_row()
        creators = _make_df(2).drop(columns=["creator_id"])
        with pytest.raises(ValueError, match="creator_id"):
            pred.predict_and_rank("ridge", camp, creators)

    def test_model_name_invalid_xgboost(self):
        pred = self._import_predict()
        camp = _base_row()
        creators = _make_df(2)
        with pytest.raises(ValueError, match="model_name"):
            pred.predict_and_rank("xgboost", camp, creators)

    def test_model_name_case_sensitive(self):
        pred = self._import_predict()
        camp = _base_row()
        creators = _make_df(2)
        with pytest.raises(ValueError, match="model_name"):
            pred.predict_and_rank("Ridge", camp, creators)

    def test_campaign_as_series_accepted(self):
        """campaign là pd.Series → to_dict, không TypeError ở bước parse."""
        pred = self._import_predict()
        camp = pd.Series(_base_row())
        creators = _make_df(1)
        try:
            pred.predict_and_rank("ridge", camp, creators)
        except (FileNotFoundError, ValueError) as e:
            if "campaign must be" in str(e):
                pytest.fail("pd.Series campaign bị từ chối")

    def test_creators_not_dataframe(self):
        pred = self._import_predict()
        camp = _base_row()
        with pytest.raises(ValueError, match="DataFrame"):
            pred.predict_and_rank("ridge", camp, [{"creator_id": "C1"}])


# ===========================================================================
# B. evaluate.py (8 cases)
# ===========================================================================

class TestEvaluateAdversarial:

    def test_single_creator_campaign_skipped_spearman(self):
        df = pd.DataFrame({
            "campaign_id": ["CP1"],
            "fit_score": [80.0],
            "pred": [75.0],
        })
        m = ranking_metrics(df, "pred", "fit_score", k=10)
        assert m["n_campaigns"] == 0
        assert np.isnan(m["spearman_mean"])

    def test_constant_fit_score_skips_spearman(self):
        df = pd.DataFrame({
            "campaign_id": ["CP1"] * 5,
            "fit_score": [50.0] * 5,
            "pred": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        m = ranking_metrics(df, "pred", "fit_score", k=3)
        assert m["n_campaigns"] == 1
        assert np.isnan(m["spearman_mean"])
        assert not np.isnan(m["ndcg@k_mean"])

    def test_k_greater_than_n_creators(self):
        df = pd.DataFrame({
            "campaign_id": ["CP1"] * 3,
            "fit_score": [90.0, 70.0, 50.0],
            "pred": [80.0, 60.0, 40.0],
        })
        m = ranking_metrics(df, "pred", "fit_score", k=100)
        assert m["k"] == 100
        assert not np.isnan(m["precision@k_mean"])
        assert not np.isnan(m["ndcg@k_mean"])

    def test_score_col_missing(self):
        df = pd.DataFrame({
            "campaign_id": ["CP1", "CP1"],
            "fit_score": [80.0, 60.0],
        })
        with pytest.raises(ValueError, match="missing column"):
            ranking_metrics(df, "pred", "fit_score", k=2)

    def test_true_col_missing(self):
        df = pd.DataFrame({
            "campaign_id": ["CP1", "CP1"],
            "pred": [80.0, 60.0],
        })
        with pytest.raises(ValueError, match="missing column"):
            ranking_metrics(df, "pred", "fit_score", k=2)

    def test_campaign_id_nan_groupby(self):
        df = pd.DataFrame({
            "campaign_id": ["CP1", "CP1", np.nan, np.nan],
            "fit_score": [90.0, 70.0, 80.0, 60.0],
            "pred": [85.0, 65.0, 75.0, 55.0],
        })
        m = ranking_metrics(df, "pred", "fit_score", k=2)
        assert m["n_campaigns"] >= 1

    def test_nan_target_validate_fails(self):
        df = _make_df(3)
        df.loc[0, "fit_score"] = np.nan
        res = validate_dataframe(df, "test_nan_target")
        assert res["ok"] is False
        assert any("target" in e.lower() or "fit_score" in e.lower() for e in res["errors"])

    def test_y_pred_nan_regression_metrics(self):
        yt = np.array([10.0, 20.0, 30.0])
        yp = np.array([11.0, np.nan, 29.0])
        try:
            m = regression_metrics(yt, yp)
            assert "MAE" in m and "RMSE" in m and "R2" in m
        except ValueError:
            pass


# ===========================================================================
# C. preprocessing.py (6 cases)
# ===========================================================================

class TestPreprocessingAdversarial:

    def test_forbidden_features_not_in_all_features(self):
        from src.preprocessing import ALL_FEATURES
        leaked = set(ALL_FEATURES) & FORBIDDEN_FEATURES
        assert leaked == set(), f"Forbidden features in ALL_FEATURES: {leaked}"

    def test_target_not_in_features(self):
        from src.preprocessing import ALL_FEATURES
        assert TARGET not in ALL_FEATURES
        assert TARGET_SOURCE not in ALL_FEATURES

    def test_missing_feature_after_add_features_raises(self):
        from src.preprocessing import transform, build_preprocessor
        df = _make_df(2)
        # drop a column that is NOT used by add_features() but IS in ALL_FEATURES
        df = df.drop(columns=["brand_safety_score"])
        pre = build_preprocessor()
        with pytest.raises(ValueError, match="Missing feature columns"):
            transform(pre, df)

    def test_ohe_handle_unknown_ignore(self):
        from src.preprocessing import build_preprocessor
        pre = build_preprocessor()
        # `pre.transformers` là list các tuple (name, transformer, columns).
        # Thuộc tính `named_transformers_` chỉ có sau khi fit, nên ta tìm trực tiếp trong list.
        cat_pipe = next(t for n, t, _ in pre.transformers if n == "cat")
        ohe = cat_pipe.named_steps["onehot"]
        assert ohe.handle_unknown == "ignore"

    def test_assert_no_forbidden_on_clean_pre(self):
        from src.preprocessing import build_preprocessor, _assert_no_forbidden
        pre = build_preprocessor()
        _assert_no_forbidden(pre)

    def test_numeric_and_categorical_disjoint_from_forbidden(self):
        from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES
        for name, feats in [("num", NUMERIC_FEATURES), ("cat", CATEGORICAL_FEATURES)]:
            bad = set(feats) & FORBIDDEN_FEATURES
            assert not bad, f"{name} contains forbidden: {bad}"


# ===========================================================================
# D. feature_engineering.py (6 cases)
# ===========================================================================

class TestFeatureEngineeringAdversarial:

    def test_min_followers_zero_no_inf(self):
        df = _make_df(1, min_followers=0, followers=5000)
        out = add_features(df)
        assert np.isnan(out["follower_ratio"].iloc[0])
        assert not np.isinf(out["follower_ratio"].iloc[0])

    def test_followers_zero_cost_and_view_nan(self):
        df = _make_df(1, followers=0, budget_usd=1000, avg_views=100)
        out = add_features(df)
        assert np.isnan(out["cost_per_follower"].iloc[0])
        assert np.isnan(out["view_rate"].iloc[0])

    def test_followers_nan_meets_min_nan(self):
        df = _make_df(1)
        df["followers"] = np.nan
        out = add_features(df)
        assert np.isnan(out["meets_min_followers"].iloc[0])

    def test_budget_zero_cost_per_follower_zero(self):
        df = _make_df(1, budget_usd=0, followers=10_000)
        out = add_features(df)
        assert out["cost_per_follower"].iloc[0] == 0.0

    def test_all_zero_no_crash(self):
        df = _make_df(1, followers=0, avg_views=0, budget_usd=0, min_followers=0)
        out = add_features(df)
        assert len(out) == 1
        for col in ["follower_ratio", "cost_per_follower", "view_rate"]:
            assert not np.isinf(out[col].iloc[0])

    def test_string_numeric_coerced_to_nan(self):
        s_num = pd.Series(["1000", "bad", "2000"])
        s_den = pd.Series([10, 10, 0])
        out = _safe_div(s_num, s_den)
        assert out.iloc[0] == 100.0
        assert np.isnan(out.iloc[1])
        assert np.isnan(out.iloc[2])


# ===========================================================================
# E. data_validation.py (6 cases)
# ===========================================================================

class TestDataValidationAdversarial:

    def test_nan_id_no_impute_warning(self):
        df = _make_df(2)
        df.loc[0, "creator_id"] = np.nan
        res = validate_dataframe(df, "nan_id")
        for w in res["warnings"]:
            assert "creator_id" not in w

    def test_nan_feature_warning_impute(self):
        df = _make_df(2)
        df.loc[0, "followers"] = np.nan
        res = validate_dataframe(df, "nan_feat")
        assert any("will be imputed" in w for w in res["warnings"])
        assert any("followers" in w for w in res["warnings"])

    def test_duplicate_creator_campaign(self):
        df = _make_df(2)
        df.loc[1, "creator_id"] = df.loc[0, "creator_id"]
        df.loc[1, "campaign_id"] = df.loc[0, "campaign_id"]
        res = validate_dataframe(df, "dup")
        assert res["ok"] is False
        assert any("duplicate" in e.lower() for e in res["errors"])

    def test_fit_score_out_of_range(self):
        df = _make_df(2, fit_score=-1.0)
        res = validate_dataframe(df, "bad_score")
        assert res["ok"] is False
        assert any("outside [0,100]" in e for e in res["errors"])

    def test_engagement_rate_out_of_range(self):
        df = _make_df(2, engagement_rate=1.5)
        res = validate_dataframe(df, "bad_eng")
        assert res["ok"] is False
        assert any("outside [0,1]" in e or "engagement_rate" in e for e in res["errors"])

    def test_invalid_target_source(self):
        df = _make_df(2, target_source="fake")
        res = validate_dataframe(df, "bad_src")
        assert res["ok"] is False
        assert any("invalid" in e.lower() and "target_source" in e for e in res["errors"])


# ===========================================================================
# Extra: overlap & contract helpers
# ===========================================================================

def test_no_campaign_overlap_clean():
    train = _make_df(2, campaign_id="CP_TRAIN")
    test = _make_df(2, campaign_id="CP_TEST")
    res = check_no_campaign_overlap(train, test)
    assert res["ok"] is True
    assert res["overlap_campaign_ids"] == []


def test_campaign_overlap_detected():
    train = _make_df(2, campaign_id="CP_SHARED")
    test = _make_df(2, campaign_id="CP_SHARED")
    res = check_no_campaign_overlap(train, test)
    assert res["ok"] is False
    assert "CP_SHARED" in res["overlap_campaign_ids"]


def test_validate_empty_dataframe():
    df = pd.DataFrame(columns=REQUIRED_COLS)
    res = validate_dataframe(df, "empty")
    assert res["ok"] is False
    assert any("empty" in e.lower() for e in res["errors"])


def test_validate_missing_required_column_raises():
    df = _make_df(1).drop(columns=["goal"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataframe(df, "missing_col")
