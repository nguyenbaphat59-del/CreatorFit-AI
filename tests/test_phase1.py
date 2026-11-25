"""Test phase 1: không rò rỉ, chia cho 0, cột cấm, range validation."""
import numpy as np
import pandas as pd
import pytest

from src.data_validation import validate_dataframe
from src.feature_engineering import add_features
from src.preprocessing import (
    fit_preprocessor, transform,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)
from src.schema import FORBIDDEN_FEATURES, TARGET


# ---------- fixtures ----------

def _make_train_df() -> pd.DataFrame:
    """Train: 3 dòng, không NaN, meets_min_followers có biến thiên [1,0,1]."""
    return pd.DataFrame({
        "creator_id":    ["C1", "C2", "C3"],
        "creator_name":  ["A", "B", "C"],
        "campaign_id":   ["CP1", "CP1", "CP1"],
        "campaign_name": ["X", "X", "X"],
        "followers":     [1000, 2000, 5000],
        "avg_views":     [100, 200, 400],
        "engagement_rate": [0.02, 0.03, 0.05],
        "posts_per_week": [5, 3, 7],
        "avg_video_length_sec": [300, 200, 400],
        "past_collabs": [3, 2, 5],
        "brand_safety_score": [80, 90, 85],
        "niche":    ["fashion", "gaming", "food"],
        "platform": ["instagram", "tiktok", "youtube"],
        "country":  ["VN", "ID", "VN"],
        "budget_usd":     [1000, 1000, 1000],
        "min_followers":  [500, 3000, 4000],   # -> meets = 1, 0, 1
        "duration_days":  [30, 30, 30],
        "campaign_niche": ["fashion", "fashion", "fashion"],
        "campaign_platform": ["instagram", "instagram", "instagram"],
        "target_country": ["VN", "VN", "VN"],
        "goal": ["awareness", "awareness", "awareness"],
        "fit_score": [70.0, 50.0, 80.0],
        "target_source": ["synthetic", "synthetic", "synthetic"],
    })


def _make_test_df() -> pd.DataFrame:
    """Test: 2 dòng, không trùng creator/campaign với train, có giá trị lạ."""
    return pd.DataFrame({
        "creator_id":    ["C10", "C11"],
        "creator_name":  ["X", "Y"],
        "campaign_id":   ["CP2", "CP2"],
        "campaign_name": ["Z", "Z"],
        "followers":     [1500, 3000],
        "avg_views":     [150, 300],
        "engagement_rate": [0.025, 0.04],
        "posts_per_week": [4, 6],
        "avg_video_length_sec": [250, 350],
        "past_collabs": [1, 4],
        "brand_safety_score": [85, 88],
        "niche":    ["fashion", "food"],
        "platform": ["instagram", "youtube"],
        "country":  ["VN", "VN"],
        "budget_usd":     [2000, 2000],
        "min_followers":  [1000, 2000],
        "duration_days":  [20, 20],
        "campaign_niche": ["beauty", "beauty"],
        "campaign_platform": ["tiktok", "tiktok"],
        "target_country": ["VN", "VN"],
        "goal": ["engagement", "engagement"],
        "fit_score": [60.0, 70.0],
        "target_source": ["synthetic", "synthetic"],
    })


# ---------- tests cũ (giữ nguyên) ----------

def test_safe_division():
    df = _make_train_df()
    df.loc[df.index[1], "followers"] = 0
    df.loc[df.index[2], "min_followers"] = 0
    out = add_features(df)
    assert pd.isna(out.loc[1, "cost_per_follower"])
    assert pd.isna(out.loc[1, "view_rate"])
    assert pd.isna(out.loc[2, "follower_ratio"])
    for col in ("follower_ratio", "cost_per_follower", "view_rate"):
        vals = out[col].fillna(0).to_numpy()
        assert not np.isinf(vals).any(), f"inf found in {col}"


def test_meets_min_followers():
    """followers >= min_followers -> 1; ngược lại 0; NaN khi thiếu; min=0 -> 1."""
    df = pd.DataFrame({
        "followers":     [1000, 1000, 1000, np.nan, 0],
        "min_followers": [500,  2000, 0,    500,    0],
    })
    # Các cột khác của add_features không cần
    for c in ("avg_views", "budget_usd"):
        df[c] = 0
    for c in ("niche", "campaign_niche", "platform", "campaign_platform",
              "country", "target_country"):
        df[c] = "x"
    out = add_features(df)
    assert out.loc[0, "meets_min_followers"] == 1
    assert out.loc[1, "meets_min_followers"] == 0
    assert out.loc[2, "meets_min_followers"] == 1   # min_followers = 0
    assert pd.isna(out.loc[3, "meets_min_followers"])
    assert out.loc[4, "meets_min_followers"] == 1   # cả hai = 0 -> 1


def test_no_forbidden_features_in_preprocessor():
    all_feats = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    assert all_feats.isdisjoint(FORBIDDEN_FEATURES)
    assert TARGET not in all_feats


def test_unknown_category_does_not_crash():
    train = _make_train_df()
    test = _make_test_df()
    test.loc[test.index[0], "niche"] = "NEVER_SEEN"
    pre, _, _ = fit_preprocessor(train)
    X = transform(pre, test)
    assert X.shape[0] == 2


# ---------- TEST LEAKAGE (viết lại) ----------

def _pandas_expected_scaler(train_feat: pd.DataFrame):
    """Mean/scale độc lập bằng pandas (median-impute + population std)."""
    means, scales = [], []
    for col in NUMERIC_FEATURES:
        v = train_feat[col].astype(float)
        imputed = v.fillna(v.median())
        means.append(imputed.mean())
        s = imputed.std(ddof=0)
        if s == 0:  # sklearn StandardScaler dùng 1.0 cho cột zero-variance
            s = 1.0
        scales.append(s)
    return np.asarray(means), np.asarray(scales)


def test_no_leakage_fit_on_train_only():
    """Train/test tách biệt hoàn toàn. Fit chỉ thấy train.

    Sau khi transform test (giá trị cực lớn + category lạ):
      * scaler.mean_, scaler.scale_ KHÔNG đổi;
      * OHE categories_ KHÔNG đổi;
      * giá trị học được BẰNG tính độc lập từ train bằng pandas.
    """
    train = _make_train_df()
    test = _make_test_df()
    assert set(train["creator_id"]).isdisjoint(test["creator_id"])
    assert set(train["campaign_id"]).isdisjoint(test["campaign_id"])

    pre, Xtr, ytr = fit_preprocessor(train)

    num_pipe = pre.named_transformers_["num"]
    scaler = num_pipe.named_steps["scaler"]
    mean_before = scaler.mean_.copy()
    scale_before = scaler.scale_.copy()

    cat_pipe = pre.named_transformers_["cat"]
    ohe = cat_pipe.named_steps["onehot"]
    cats_before = [c.copy() for c in ohe.categories_]

    # Transform test: giá trị cực lớn + category lạ
    test_mod = test.copy()
    test_mod.loc[test_mod.index[0], "followers"] = 1e12
    test_mod.loc[test_mod.index[0], "engagement_rate"] = 0.999
    test_mod.loc[test_mod.index[0], "niche"] = "NEVER_SEEN_NICHE"
    test_mod.loc[test_mod.index[0], "platform"] = "NEVER_SEEN_PLATFORM"
    _ = transform(pre, test_mod)

    # (a) Tham số KHÔNG đổi
    assert np.allclose(scaler.mean_, mean_before), "scaler.mean_ changed!"
    assert np.allclose(scaler.scale_, scale_before), "scaler.scale_ changed!"
    for cb, ca in zip(cats_before, ohe.categories_):
        assert np.array_equal(cb, ca), "OHE categories_ changed!"

    # (b) Bằng giá trị tính độc lập bằng pandas từ train
    train_feat = add_features(train)
    exp_mean, exp_scale = _pandas_expected_scaler(train_feat)
    assert np.allclose(mean_before, exp_mean, atol=1e-9), (
        f"scaler.mean_ mismatch:\n got={mean_before}\n exp={exp_mean}"
    )
    assert np.allclose(scale_before, exp_scale, atol=1e-9), (
        f"scaler.scale_ mismatch:\n got={scale_before}\n exp={exp_scale}"
    )

    # (c) OHE categories = unique values của train (sorted)
    for j, col in enumerate(CATEGORICAL_FEATURES):
        exp_cats = np.sort(train_feat[col].dropna().astype(str).unique())
        got = np.asarray(cats_before[j], dtype=object)
        assert np.array_equal(np.asarray(got.astype(str)), exp_cats), (
            f"categories_[{j}] for '{col}': got={got}, exp={exp_cats}"
        )


# ---------- TEST VALIDATION RANGE (mỗi điều kiện phải bị bắt) ----------

@pytest.mark.parametrize("col, bad_value", [
    ("engagement_rate", 1.5),
    ("engagement_rate", -0.1),
    ("brand_safety_score", 150.0),
    ("brand_safety_score", -1.0),
    ("followers", -1.0),
    ("avg_views", -1.0),
    ("budget_usd", -1.0),
    ("min_followers", -1.0),
    ("duration_days", -1.0),
    ("posts_per_week", -1.0),
])
def test_validation_catches_range_violations(col, bad_value):
    df = _make_train_df()
    df.loc[df.index[0], col] = bad_value
    result = validate_dataframe(df, name="test")
    assert not result["ok"], f"expected error for {col}={bad_value}"
    assert any(col in e for e in result["errors"]), (
        f"no error mentioned '{col}': {result['errors']}"
    )


def test_validation_passes_clean_data():
    result = validate_dataframe(_make_train_df(), name="train")
    assert result["ok"], result["errors"]


def test_imputation_warning_excludes_ids():
    """NaN ở ID KHÔNG được warn; NaN ở feature THÌ được warn."""
    df = _make_train_df()
    df.loc[df.index[0], "followers"] = np.nan       # feature
    df.loc[df.index[0], "creator_name"] = np.nan    # ID
    df.loc[df.index[0], "campaign_name"] = np.nan   # ID
    result = validate_dataframe(df, name="test")
    warn_text = " | ".join(result["warnings"])
    assert "followers" in warn_text, f"missing warning for followers: {warn_text}"
    assert "creator_name" not in warn_text, f"unexpected warn for ID: {warn_text}"
    assert "campaign_name" not in warn_text, f"unexpected warn for ID: {warn_text}"
