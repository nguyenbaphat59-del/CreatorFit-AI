"""Ba mô hình: Ridge, RandomForest, MLP PyTorch (CPU-friendly)."""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
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


# =========================================================
# MLP (PyTorch, CPU)
# =========================================================
class MLPRegressorTorch:
    """MLP 2–3 lớp ẩn, ReLU, Dropout, AdamW, MSE, early stopping.

    - `fit(X, y, groups=...)`: nếu có `groups` (campaign_id) thì tách validation
      THEO campaign_id; nếu không, tách ngẫu nhiên (không khuyến nghị).
    - `predict(X) -> np.ndarray`
    - `save(path)` / `MLPRegressorTorch.load(path)`
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        patience: int = 15,
        batch_size: int = 64,
        val_fraction: float = 0.2,
        random_state: int = RANDOM_STATE,
        device: str = "cpu",
    ) -> None:
        if not (2 <= len(hidden_dims) <= 3):
            raise ValueError("hidden_dims phải có 2–3 lớp ẩn.")
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.val_fraction = float(val_fraction)
        self.random_state = int(random_state)
        self.device = torch.device(device)

        self.model_: Optional[nn.Module] = None
        self.input_dim_: Optional[int] = None
        self.n_epochs_: int = 0
        self.best_val_loss_: float = float("inf")

    # ---- build ----
    def _build(self, in_dim: int) -> nn.Module:
        layers: list[nn.Module] = []
        prev = in_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers).to(self.device)

    # ---- split val theo campaign_id ----
    @staticmethod
    def _split_by_campaign(
        groups: np.ndarray, val_fraction: float, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        unique = np.array(sorted(np.unique(groups)))
        rng = np.random.default_rng(seed)
        rng.shuffle(unique)
        n_val = max(1, int(round(len(unique) * val_fraction)))
        n_val = min(n_val, len(unique) - 1)  # đảm bảo còn ≥1 campaign train
        val_groups = set(unique[:n_val].tolist())
        val_mask = np.array([g in val_groups for g in groups])
        return np.where(~val_mask)[0], np.where(val_mask)[0]

    # ---- fit ----
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "MLPRegressorTorch":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        self.input_dim_ = X.shape[1]
        self.model_ = self._build(self.input_dim_)

        # tách validation
        if X_val is not None and y_val is not None:
            X_tr, y_tr = X, y
            X_va = np.asarray(X_val, dtype=np.float32)
            y_va = np.asarray(y_val, dtype=np.float32).ravel()
        else:
            if groups is not None:
                tr_idx, va_idx = self._split_by_campaign(
                    np.asarray(groups), self.val_fraction, self.random_state
                )
            else:
                n = len(X)
                rng = np.random.default_rng(self.random_state)
                idx = rng.permutation(n)
                n_val = max(1, int(round(n * self.val_fraction)))
                va_idx, tr_idx = idx[:n_val], idx[n_val:]
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]

        X_tr_t = torch.from_numpy(X_tr).to(self.device)
        y_tr_t = torch.from_numpy(y_tr).to(self.device).unsqueeze(1)
        X_va_t = torch.from_numpy(X_va).to(self.device)
        y_va_t = torch.from_numpy(y_va).to(self.device).unsqueeze(1)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        n_train = X_tr_t.shape[0]
        best_state = None
        best_val = float("inf")
        epochs_no_improve = 0
        self.n_epochs_ = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            perm = torch.randperm(n_train, device=self.device)
            for start in range(0, n_train, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb, yb = X_tr_t[idx], y_tr_t[idx]
                optimizer.zero_grad()
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()

            # validation
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(X_va_t), y_va_t).item())

            self.n_epochs_ = epoch
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                epochs_no_improve = 0
                best_state = {
                    k: v.detach().clone() for k, v in self.model_.state_dict().items()
                }
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.best_val_loss_ = float(best_val)
        return self

    # ---- predict ----
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("MLPRegressorTorch chưa được fit.")
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            x = torch.from_numpy(X).to(self.device)
            out = self.model_(x).cpu().numpy().ravel()
        return out.astype(np.float64)

    # ---- persist ----
    def state_dict_with_meta(self) -> dict:
        return {
            "config": {
                "hidden_dims": list(self.hidden_dims),
                "dropout": self.dropout,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "batch_size": self.batch_size,
                "val_fraction": self.val_fraction,
                "random_state": self.random_state,
            },
            "input_dim": int(self.input_dim_),
            "state_dict": self.model_.state_dict(),
            "n_epochs": int(self.n_epochs_),
            "best_val_loss": float(self.best_val_loss_),
        }

    @classmethod
    def load(cls, path, device: str = "cpu") -> "MLPRegressorTorch":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        obj = cls(device=device, **ckpt["config"])
        obj.model_ = obj._build(int(ckpt["input_dim"]))
        obj.model_.load_state_dict(ckpt["state_dict"])
        obj.model_.eval()
        obj.input_dim_ = int(ckpt["input_dim"])
        obj.n_epochs_ = int(ckpt.get("n_epochs", 0))
        obj.best_val_loss_ = float(ckpt.get("best_val_loss", float("inf")))
        return obj


if __name__ == "__main__":  # demo nhỏ
    rng = np.random.default_rng(RANDOM_STATE)
    X = rng.normal(size=(60, 5))
    y = X[:, 0] * 3.0 + X[:, 1] * -1.5 + rng.normal(scale=0.1, size=60)
    groups = np.repeat(np.arange(6), 10)

    a, scores = tune_ridge(X, y, groups)
    print(f"[demo] best alpha={a} | RMSE={scores[a]:.4f}")

    params, info = tune_random_forest(X, y, groups)
    print(f"[demo] best rf params={params} | RMSE={info['best_score']:.4f}")

    mlp = MLPRegressorTorch(max_epochs=50, patience=5)
    mlp.fit(X, y, groups=groups)
    print(
        f"[demo] mlp epochs={mlp.n_epochs_} | best_val_mse={mlp.best_val_loss_:.4f} "
        f"| pred[:3]={mlp.predict(X[:3]).round(2)}"
    )
