"""Xuất kết quả xếp hạng CSV/XLSX."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
import pandas as pd


def to_csv(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); df.to_csv(p, index=False); return p


def to_excel(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); df.to_excel(p, index=False); return p


def export_ranking(df: pd.DataFrame, campaign_id: str, model_name: str, out_dir: str | Path = "outputs") -> dict[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"ranking_{campaign_id}_{model_name}_{ts}"
    out = Path(out_dir)
    return {"csv": str(to_csv(df, out / f"{base}.csv")),
            "xlsx": str(to_excel(df, out / f"{base}.xlsx"))}


def ranking_filename(campaign_id: str, model_name: str, ext: str, ts: str | None = None) -> str:
    """Tên file chuẩn: ranking_<campaign_id>_<model>_<timestamp>.<ext>."""
    ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"ranking_{campaign_id}_{model_name}_{ts}.{ext}"


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV (UTF-8 có BOM để Excel đọc đúng tiếng Việt) - không ghi ra đĩa."""
    return df.to_csv(index=False).encode("utf-8-sig")


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    """XLSX trong bộ nhớ - không ghi ra đĩa."""
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


if __name__ == "__main__":
    demo = pd.DataFrame({"rank": [1, 2], "creator_id": ["C1", "C2"],
                         "creator_name": ["A", "B"], "fit_score_pred": [88.0, 70.0]})
    print(export_ranking(demo, campaign_id="CP_DEMO", model_name="ridge"))