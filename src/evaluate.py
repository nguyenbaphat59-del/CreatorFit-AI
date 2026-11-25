"""Đánh giá 3 mô hình trên test.csv. CHỈ transform — KHÔNG fit trên test."""
from __future__ import annotations

from pathlib import Path

import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data_validation import validate_dataframe
from src.models import MLPRegressorTorch
from src.preprocessing import transform

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, MODELS_DIR, OUT_DIR = ROOT / "data", ROOT / "models", ROOT / "outputs"


def regression_metrics(y_true, y_pred) -> dict:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if len(yt) != len(yp):
        raise ValueError("length mismatch")
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "R2": float(r2_score(yt, yp)) if len(yt) > 1 else float("nan"),
    }


def _dcg(rel: np.ndarray, k: int) -> float:
    r = np.asarray(rel, dtype=float)[:k]
    if len(r) == 0:
        return 0.0
    pos = np.arange(1, len(r) + 1, dtype=float)
    return float(np.sum(r / np.log2(pos + 1.0)))


def _ndcg_at_k(true: np.ndarray, pred: np.ndarray, k: int) -> float:
    n = len(true); kk = min(int(k), n)
    if kk <= 0:
        return float("nan")
    true = np.asarray(true, dtype=float); pred = np.asarray(pred, dtype=float)
    idcg = _dcg(true[np.argsort(-true, kind="stable")], kk)
    dcg = _dcg(true[np.argsort(-pred, kind="stable")], kk)
    return float(dcg / idcg) if idcg > 0 else 0.0


def _precision_at_k(true: np.ndarray, pred: np.ndarray, k: int) -> float:
    n = len(true); kk = min(int(k), n)
    if kk <= 0:
        return float("nan")
    tp = set(np.argsort(-np.asarray(pred, dtype=float), kind="stable")[:kk].tolist())
    tt = set(np.argsort(-np.asarray(true, dtype=float), kind="stable")[:kk].tolist())
    return float(len(tp & tt) / kk)


def ranking_metrics(df: pd.DataFrame, score_col: str, true_col: str = "fit_score", k: int = 10) -> dict:
    for c in ("campaign_id", score_col, true_col):
        if c not in df.columns:
            raise ValueError(f"missing column {c!r}")

    sp, nd, ps = [], [], []
    n_eval = 0
    for _, g in df.groupby("campaign_id", sort=False):
        if len(g) < 2:
            continue
        n_eval += 1
        t = g[true_col].to_numpy(dtype=float)
        p = g[score_col].to_numpy(dtype=float)
        kk = min(k, len(g))
        if not (np.all(p == p[0]) or np.all(t == t[0])):
            v, _ = spearmanr(t, p)
            if not np.isnan(v):
                sp.append(float(v))
        nd.append(_ndcg_at_k(t, p, kk))
        ps.append(_precision_at_k(t, p, kk))

    def ms(xs): return (float(np.mean(xs)), float(np.std(xs))) if xs else (float("nan"),) * 2

    sp_m, sp_s = ms(sp); nd_m, nd_s = ms(nd); p_m, p_s = ms(ps)
    return {"spearman_mean": sp_m, "spearman_std": sp_s,
            "ndcg@k_mean": nd_m, "ndcg@k_std": nd_s,
            "precision@k_mean": p_m, "precision@k_std": p_s,
            "n_campaigns": n_eval, "k": k}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print(f"[evaluate] test.csv shape={test_df.shape}")

    vr = validate_dataframe(test_df, name="test.csv")
    print(f"[evaluate] validate ok={vr['ok']} n_rows={vr['n_rows']}")
    for m in vr["warnings"] + vr["errors"]:
        print(f"[evaluate] {m}")
    if not vr["ok"]:
        raise SystemExit("test.csv invalid — dừng.")

    pre = joblib.load(MODELS_DIR / "preprocessor.joblib")
    models = {
        "ridge": joblib.load(MODELS_DIR / "ridge.joblib"),
        "rf": joblib.load(MODELS_DIR / "rf.joblib"),
        "mlp": MLPRegressorTorch.load(MODELS_DIR / "mlp.pt"),
    }

    X_test = transform(pre, test_df)  # CHỈ transform
    sources = sorted(test_df["target_source"].dropna().unique().tolist())
    has_synthetic = "synthetic" in sources

    rows, preds = [], {}
    for name, model in models.items():
        yp = np.clip(np.asarray(model.predict(X_test), dtype=float).ravel(), 0.0, 100.0)
        preds[name] = yp
        reg = regression_metrics(test_df["fit_score"], yp)
        rank_df = test_df[["campaign_id", "fit_score"]].copy()
        rank_df["pred"] = yp
        rk = ranking_metrics(rank_df, "pred", "fit_score", k=10)
        rows.append({"model": name, "MAE": reg["MAE"], "RMSE": reg["RMSE"], "R2": reg["R2"],
                     "spearman_mean": rk["spearman_mean"], "spearman_std": rk["spearman_std"],
                     "ndcg@10_mean": rk["ndcg@k_mean"], "p@10_mean": rk["precision@k_mean"],
                     "n_campaigns": rk["n_campaigns"]})

    mdf = pd.DataFrame(rows)[["model", "MAE", "RMSE", "R2", "spearman_mean", "spearman_std",
                              "ndcg@10_mean", "p@10_mean", "n_campaigns"]]
    mdf.to_csv(OUT_DIR / "metrics.csv", index=False)
    print(f"[evaluate] wrote {OUT_DIR/'metrics.csv'}")

    best_rmse = mdf.loc[mdf["RMSE"].idxmin()]
    best_sp = mdf.loc[mdf["spearman_mean"].idxmax()] if mdf["spearman_mean"].notna().any() else None

    lines = ["# Evaluation Report — CreatorFit AI", "",
             "## Test set",
             f"- Rows: {len(test_df)}",
             f"- Campaigns: {test_df['campaign_id'].nunique()}",
             f"- target_source: {sources}", ""]
    if has_synthetic:
        lines += ["> **Kết quả trên dữ liệu synthetic chỉ xác nhận pipeline, không phản ánh hiệu năng thực.**", ""]
    cols = ["model", "MAE", "RMSE", "R2", "spearman_mean", "spearman_std",
            "ndcg@10_mean", "p@10_mean", "n_campaigns"]
    lines += ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in mdf.iterrows():
        lines += ["| " + " | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |"]
    lines += ["", "## Best model (từ số liệu chạy ra — không thiên vị mô hình)",
              f"- RMSE thấp nhất: **{best_rmse['model']}** ({best_rmse['RMSE']:.4f})"]
    if best_sp is not None:
        lines += [f"- Spearman@10 cao nhất: **{best_sp['model']}** ({best_sp['spearman_mean']:.4f})"]
    (OUT_DIR / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[evaluate] wrote {OUT_DIR/'evaluation_report.md'}")

    true = test_df["fit_score"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), squeeze=False)
    for ax, (name, yp) in zip(axes[0], preds.items()):
        ax.scatter(true, yp, s=8, alpha=0.5); ax.plot([0, 100], [0, 100], "r--", lw=0.8)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_xlabel("fit_score true"); ax.set_ylabel("fit_score_pred"); ax.set_title(name)
    fig.tight_layout(); fig.savefig(OUT_DIR / "pred_vs_true.png", dpi=120); plt.close(fig)
    print(f"[evaluate] wrote {OUT_DIR/'pred_vs_true.png'}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (col, title) in zip(axes.ravel(), [
        ("MAE", "MAE (thấp = tốt)"), ("RMSE", "RMSE (thấp = tốt)"),
        ("R2", "R² (cao = tốt)"), ("spearman_mean", "Spearman@10 (cao = tốt)")]):
        ax.bar(mdf["model"], mdf[col]); ax.set_title(title); ax.set_xlabel("model")
        ax.set_ylabel(col); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / "metric_compare.png", dpi=120); plt.close(fig)
    print(f"[evaluate] wrote {OUT_DIR/'metric_compare.png'}")
    print("[evaluate] DONE.")


if __name__ == "__main__":
    main()