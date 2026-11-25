"""Mô hình baseline: Ridge và RandomForest (MLP PyTorch sẽ được bổ sung ở PR sau)."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold

from src.schema import RANDOM_STATE

# =========================================================
# Ridge
# =========================================================
def get_ridge(alpha: float = 1.0) -> Ridge:
    """Trả về Ridge đã cấu hình. Ridge KHÔNG có random_state (deterministic)."""
    return Ridge(alpha=float(alpha))


def tune_ridge(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: Iterable[float] = (0.01, 0.1, 1.0, 10.0, 30.0, 100.0, 300.0),
    n_splits: int = 5,
) -> tuple[float, dict[float, float]]:
    """Chọn alpha bằng GroupKFold theo `campaign_id`, CHỈ trên TRAIN.

    Trả về (best_alpha, {alpha: mean_rmse}).
    Raise ValueError nếu số campaign < 2.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float).ravel()
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError(
            f"tune_ridge: cần ≥ 2 campaign_id để GroupKFold, hiện có {len(unique_groups)}"
        )
    k = min(n_splits, len(unique_groups))
    gkf = GroupKFold(n_splits=k)

    scores: dict[float, float] = {}
    best_alpha, best_rmse = None, float("inf")

    for a in alphas:
        fold_rmse: list[float] = []
        for tr_idx, va_idx in gkf.split(X, y, groups):
            model = get_ridge(alpha=a)
            model.fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[va_idx])
            fold_rmse.append(float(np.sqrt(mean_squared_error(y[va_idx], pred))))
        mean_rmse = float(np.mean(fold_rmse))
        scores[float(a)] = mean_rmse
        if mean_rmse < best_rmse:
            best_rmse, best_alpha = mean_rmse, float(a)

    return best_alpha, scores


# =========================================================
# RandomForest
# =========================================================
def get_random_forest(
    n_estimators: int = 200,
    max_depth: Optional[int] = 20,
    min_samples_leaf: int = 1,
    random_state: int = RANDOM_STATE,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        min_samples_leaf=int(min_samples_leaf),
        random_state=random_state,
        n_jobs=-1,
    )


def tune_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 3,
) -> tuple[dict, dict]:
    """Grid nhỏ + GroupKFold (theo campaign_id), CHỈ trên TRAIN.

    Grid:
      n_estimators ∈ {100, 200}
      max_depth     ∈ {10, None}
      min_samples_leaf ∈ {1, 5}
    Trả về (best_params, {'best_score': rmse, 'all_scores': [...]}).
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float).ravel()
    groups = np.asarray(groups)

    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        # fallback: không tune được GroupKFold, dùng mặc định
        return (
            {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 1},
            {"best_score": float("nan"), "all_scores": []},
        )
    k = min(n_splits, len(unique_groups))
    gkf = GroupKFold(n_splits=k)

    grid = [
        {"n_estimators": n, "max_depth": d, "min_samples_leaf": m}
        for n in (100, 200)
        for d in (10, None)
        for m in (1, 5)
    ]

    all_scores: list[dict] = []
    best_params, best_rmse = None, float("inf")

    for params in grid:
        fold_rmse: list[float] = []
        for tr_idx, va_idx in gkf.split(X, y, groups):
            model = get_random_forest(**params)
            model.fit(X[tr_idx], y[tr_idx])
            pred = model.predict(X[va_idx])
            fold_rmse.append(float(np.sqrt(mean_squared_error(y[va_idx], pred))))
        mean_rmse = float(np.mean(fold_rmse))
        all_scores.append({"params": params, "rmse": mean_rmse})
        if mean_rmse < best_rmse:
            best_rmse, best_params = mean_rmse, params

    return best_params, {"best_score": best_rmse, "all_scores": all_scores}


if __name__ == "__main__":  # demo nhỏ
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(60, 5))
    y = X[:, 0] * 3.0 + X[:, 1] * -1.5 + rng.normal(scale=0.1, size=60)
    groups = np.repeat(np.arange(6), 10)

    a, scores = tune_ridge(X, y, groups)
    print(f"[demo] best alpha={a} | RMSE={scores[a]:.4f}")

    params, info = tune_random_forest(X, y, groups)
    print(f"[demo] best rf params={params} | RMSE={info['best_score']:.4f}")
