"""Sinh danh sách Creator mẫu (SYNTHETIC) để thử chức năng upload CSV của app.

Chạy: python scripts/make_sample_creators.py [--n 50] [--out data/sample_creators_50.csv] [--seed 2026]
File chỉ chứa thuộc tính Creator (không có cột Campaign hay fit_score): app sẽ ghép với Campaign đang chọn.
Cột target_source = synthetic để app hiện cảnh báo dữ liệu tổng hợp; cột này KHÔNG được dùng làm feature.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import schema as S  # noqa: E402
from make_synthetic_data import make_creators  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "sample_creators_50.csv"))
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    df = make_creators(np.random.default_rng(a.seed), a.n)
    df["creator_id"] = [f"TEST{i:03d}" for i in range(1, a.n + 1)]
    df["creator_name"] = [f"Sample Creator {i:02d}" for i in range(1, a.n + 1)]
    df["target_source"] = "synthetic"
    cols = ["creator_id", "creator_name"] + S.CREATOR_NUMERIC + S.CREATOR_CATEGORICAL + ["target_source"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df[cols].to_csv(a.out, index=False)
    print(f"Đã tạo {a.n} Creator (synthetic) -> {a.out}")


if __name__ == "__main__":
    main()
