"""Dự đoán + xếp hạng Creator TRONG MỘT Campaign. Chỉ transform, không fit."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import transform
from src.schema import CAMPAIGN_CATEGORICAL, CAMPAIGN_NUMERIC

MODELS_DIR: Path = Path(__file__).resolve().parent.parent / "models"
_VALID_MODELS: tuple[str, ...] = ("ridge", "rf", "mlp")
_CACHE: dict[str, Any] = {}
_CAMPAIGN_REQ = tuple(CAMPAIGN_NUMERIC) + tuple(CAMPAIGN_CATEGORICAL)


def _clear_cache() -> None:
    _CACHE.clear()


def _get_preprocessor():
    if "preprocessor" not in _CACHE:
        p = MODELS_DIR / "preprocessor.joblib"
        if not p.exists():
            raise FileNotFoundError(f"preprocessor not found: {p}")
        _CACHE["preprocessor"] = joblib.load(p)
    return _CACHE["preprocessor"]


def _get_model(name: str):
    if name not in _VALID_MODELS:
        raise ValueError(f"Invalid model_name {name!r}; expected {_VALID_MODELS}")
    if name in _CACHE:
        return _CACHE[name]
    if name == "ridge":
        _CACHE[name] = joblib.load(MODELS_DIR / "ridge.joblib")
    elif name == "rf":
        _CACHE[name] = joblib.load(MODELS_DIR / "rf.joblib")
    else:
        from src.models import MLPRegressorTorch  # nạp lười: tránh import torch khi chỉ dùng ridge/rf

        _CACHE[name] = MLPRegressorTorch.load(MODELS_DIR / "mlp.pt")
    return _CACHE[name]


def predict_and_rank(
    model_name: str,
    campaign: dict | pd.Series,
    creators: pd.DataFrame,
) -> pd.DataFrame:
    if model_name not in _VALID_MODELS:
        raise ValueError(f"model_name must be one of {_VALID_MODELS}, got {model_name!r}")
    if not isinstance(creators, pd.DataFrame):
        raise ValueError("creators must be a pandas.DataFrame")
    if len(creators) == 0:
        raise ValueError("creators is empty")
    if "creator_id" not in creators.columns:
        raise ValueError("creators must contain 'creator_id'")

    camp = campaign.to_dict() if isinstance(campaign, pd.Series) else dict(campaign) if isinstance(campaign, dict) else None
    if camp is None:
        raise ValueError("campaign must be dict or pandas.Series")

    missing = [c for c in _CAMPAIGN_REQ if c not in camp]
    if missing:
        raise ValueError(f"campaign missing required fields: {missing}")

    df = creators.copy()
    uniq = df["campaign_id"].dropna().unique().tolist() if "campaign_id" in df.columns else []
    if len(uniq) > 1:
        raise ValueError(f"creators contains multiple campaign_ids: {uniq}")

    camp_cid = camp.get("campaign_id") or (uniq[0] if uniq else None)
    if camp_cid is None:
        raise ValueError("cannot determine campaign_id")
    if uniq and uniq[0] != camp_cid:
        raise ValueError(f"campaign_id mismatch: {uniq[0]!r} vs {camp_cid!r}")

    df["campaign_id"] = camp_cid
    df["campaign_name"] = camp.get("campaign_name", "")
    for col in _CAMPAIGN_REQ:
        df[col] = camp[col]
    if "creator_name" not in df.columns:
        df["creator_name"] = df["creator_id"].astype(str)

    pre = _get_preprocessor()
    model = _get_model(model_name)
    pred = np.clip(np.asarray(model.predict(transform(pre, df)), dtype=float).ravel(), 0.0, 100.0)

    out = pd.DataFrame({
        "creator_id": df["creator_id"].to_numpy(),
        "creator_name": df["creator_name"].to_numpy(),
        "fit_score_pred": pred,
    }).sort_values("fit_score_pred", ascending=False, kind="mergesort").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1, dtype=int))
    return out


def rank_all_models(campaign, creators) -> dict[str, pd.DataFrame]:
    return {n: predict_and_rank(n, campaign, creators) for n in _VALID_MODELS}