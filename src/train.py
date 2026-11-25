"""Huấn luyện 3 mô hình. TUYỆT ĐỐI KHÔNG đọc test.csv."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_validation import validate_dataframe
from src.models import (
    MLPRegressorTorch,
    get_random_forest,
    get_ridge,
    tune_random_forest,
    tune_ridge,
)
from src.preprocessing import fit_preprocessor
from src.schema import RANDOM_STATE

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


def _log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CreatorFit AI – train (Phase 2)")
    parser.add_argument("--train", default=str(DATA_DIR / "train.csv"))
    parser.add_argument("--out", default=str(MODELS_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Load + validate TRAIN ----------
    train_df = pd.read_csv(args.train)
    _log(f"loaded train.csv: shape={train_df.shape}")

    vr = validate_dataframe(train_df, name="train.csv")
    _log(f"validate: ok={vr['ok']} n_rows={vr['n_rows']}")
    for w in vr["warnings"]:
        _log(f"warning: {w}")
    for e in vr["errors"]:
        _log(f"ERROR: {e}")
    if not vr["ok"]:
        raise SystemExit("Validation TRAIN thất bại — dừng.")

    groups = train_df["campaign_id"].to_numpy()
    n_campaigns = len(np.unique(groups))
    _log(f"n_campaigns in TRAIN = {n_campaigns}")

    # ---------- 2. Preprocessing fit CHỈ trên TRAIN ----------
    t0 = time.time()
    pre, X_train, y_train = fit_preprocessor(train_df)
    dt_pre = time.time() - t0
    _log(f"preprocessor fitted: X={X_train.shape} y={y_train.shape} | {dt_pre:.2f}s")

    # ---------- 3. Ridge ----------
    _log("tuning Ridge alpha via GroupKFold (campaign_id, TRAIN only) ...")
    t0 = time.time()
    best_alpha, cv_scores = tune_ridge(X_train, y_train, groups)
    dt_tune = time.time() - t0
    _log(
        f"ridge best alpha={best_alpha} "
        f"cv_rmse_train={cv_scores[best_alpha]:.4f} | {dt_tune:.2f}s"
    )
    ridge = get_ridge(alpha=best_alpha)
    t0 = time.time()
    ridge.fit(X_train, y_train)
    _log(f"ridge fit done | {time.time()-t0:.2f}s")
    ridge_path = out_dir / "ridge.joblib"
    joblib.dump(ridge, ridge_path)
    _log(f"saved {ridge_path}")

    # ---------- 4. RandomForest ----------
    _log("tuning RandomForest (grid nhỏ, GroupKFold campaign_id, TRAIN only) ...")
    t0 = time.time()
    rf_params, rf_info = tune_random_forest(X_train, y_train, groups)
    dt_tune_rf = time.time() - t0
    _log(
        f"rf best params={rf_params} "
        f"cv_rmse_train={rf_info['best_score']:.4f} | {dt_tune_rf:.2f}s"
    )
    rf = get_random_forest(**rf_params)
    t0 = time.time()
    rf.fit(X_train, y_train)
    _log(f"rf fit done | {time.time()-t0:.2f}s")
    rf_path = out_dir / "rf.joblib"
    joblib.dump(rf, rf_path, compress=3)  # nén: file nhỏ ~4x, tải thêm ~0.04s
    _log(f"saved {rf_path}")

    # ---------- 5. MLP ----------
    _log("training MLP (val split by campaign_id, early stopping, CPU) ...")
    t0 = time.time()
    mlp = MLPRegressorTorch(random_state=RANDOM_STATE, device="cpu")
    mlp.fit(X_train, y_train, groups=groups)
    dt_mlp = time.time() - t0
    _log(
        f"mlp fit done | epochs={mlp.n_epochs_} "
        f"best_val_mse={mlp.best_val_loss_:.4f} | {dt_mlp:.2f}s"
    )
    mlp_path = out_dir / "mlp.pt"
    import torch

    torch.save(mlp.state_dict_with_meta(), mlp_path)
    _log(f"saved {mlp_path}")

    # ---------- 6. Preprocessor ----------
    pre_path = out_dir / "preprocessor.joblib"
    joblib.dump(pre, pre_path)
    _log(f"saved {pre_path}")

    _log("DONE.")


if __name__ == "__main__":
    main()
