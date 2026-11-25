"""Cổng kiểm tra tự động theo giai đoạn. Chạy: python scripts/gate.py <1..5|all>

In kết quả ngắn gọn; dán nguyên output cho điều phối viên thay vì dán code.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
R = []


def chk(name, ok, detail=""):
    R.append((bool(ok), name, detail))


def exists(*paths):
    miss = [p for p in paths if not (ROOT / p).exists()]
    chk("files: " + ",".join(paths), not miss, f"thiếu {miss}" if miss else "")
    return not miss


def txt(p):
    f = ROOT / p
    return f.read_text(encoding="utf-8") if f.exists() else ""


def code_only(p):
    """Mã nguồn đã bỏ docstring và comment (tránh dương tính giả)."""
    import ast, io, tokenize
    t = txt(p)
    try:
        tree = ast.parse(t)
        for n in ast.walk(tree):
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                b = n.body
                if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
                        and isinstance(b[0].value.value, str):
                    b[0].value.value = ""
        t = ast.unparse(tree)
    except SyntaxError:
        pass
    return t


def forbid(p, pattern, why):
    hit = re.search(pattern, code_only(p))
    chk(f"{p}: {why}", not hit, f"vi phạm: {hit.group(0)}" if hit else "")


def pytest(target):
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", target],
                       cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout.strip().splitlines() or ["(không output)"])[-1]
    chk(f"pytest {target}", r.returncode == 0, tail)


def phase1():
    if not exists("src/schema.py", "src/data_validation.py", "src/feature_engineering.py",
                  "src/preprocessing.py", "tests/test_phase1.py", "data/train.csv", "data/test.csv"):
        return
    pytest("tests/test_phase1.py")
    import pandas as pd
    from src.data_validation import validate_dataframe, check_no_campaign_overlap
    from src.preprocessing import fit_preprocessor, transform, ALL_FEATURES
    from src.schema import FORBIDDEN_FEATURES
    tr, te = pd.read_csv(ROOT / "data/train.csv"), pd.read_csv(ROOT / "data/test.csv")
    chk("validate train/test", validate_dataframe(tr, "train")["ok"] and validate_dataframe(te, "test")["ok"])
    chk("train/test không chung campaign", check_no_campaign_overlap(tr, te)["ok"])
    chk("không cột cấm trong feature", not (set(ALL_FEATURES) & FORBIDDEN_FEATURES))
    pre, X, y = fit_preprocessor(tr)
    Xt = transform(pre, te)
    import numpy as np
    chk("X không NaN, cùng số cột", not np.isnan(X).any() and not np.isnan(Xt).any() and X.shape[1] == Xt.shape[1],
        f"{X.shape} vs {Xt.shape}")


def phase2():
    if not exists("src/models.py", "src/train.py"):
        return
    forbid("src/train.py", r"test\.csv", "không được dùng test.csv")
    forbid("src/train.py", r"read_csv\([^)]*test", "không đọc test")
    chk("train.py dùng GroupKFold/campaign_id", "campaign_id" in txt("src/train.py") or "GroupKFold" in txt("src/train.py"))
    r = subprocess.run([sys.executable, "-m", "src.train"], cwd=ROOT, capture_output=True, text=True)
    chk("chạy src.train", r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1])
    exists("models/ridge.joblib", "models/rf.joblib", "models/mlp.pt", "models/preprocessor.joblib")


def phase3():
    if not exists("src/evaluate.py", "src/predict.py"):
        return
    for p in ("src/evaluate.py", "src/predict.py"):
        forbid(p, r"\.fit(_transform)?\(", "không fit trong evaluate/predict")
    r = subprocess.run([sys.executable, "-m", "src.evaluate"], cwd=ROOT, capture_output=True, text=True)
    chk("chạy src.evaluate", r.returncode == 0, (r.stderr.strip().splitlines() or [""])[-1])
    if exists("outputs/metrics.csv", "outputs/evaluation_report.md"):
        import pandas as pd
        m = pd.read_csv(ROOT / "outputs/metrics.csv")
        chk("metrics đủ 3 mô hình", m.shape[0] >= 3, f"{m.shape}")
        chk("báo cáo có cảnh báo synthetic", "synthetic" in txt("outputs/evaluation_report.md").lower())
    try:
        import pandas as pd
        from src.predict import predict_and_rank
        te = pd.read_csv(ROOT / "data/test.csv")
        cid = te.campaign_id.iloc[0]
        sub = te[te.campaign_id == cid]
        camp = sub.iloc[0]
        for mn in ("ridge", "rf", "mlp"):
            out = predict_and_rank(mn, camp, sub)
            col = [c for c in out.columns if "pred" in c][0]
            ok = out[col].between(0, 100).all() and out[col].is_monotonic_decreasing and len(out) == len(sub)
            chk(f"predict_and_rank[{mn}] 0-100, giảm dần", ok)
        try:
            predict_and_rank("ridge", te.iloc[0], te)
            chk("từ chối nhiều campaign", False, "không raise ValueError")
        except ValueError:
            chk("từ chối nhiều campaign", True)
    except Exception as e:
        chk("predict_and_rank", False, f"{type(e).__name__}: {e}")


def phase4():
    if not exists("app.py", "src/export.py"):
        return
    a = txt("app.py")
    chk("app gọi predict_and_rank", "predict_and_rank" in a)
    forbid("app.py", r"\.fit(_transform)?\(", "không huấn luyện trong app")
    chk("app có CSV + Excel export", "download_button" in a and ("xlsx" in a or "to_excel" in a))
    chk("app có cảnh báo synthetic", "synthetic" in a.lower())
    r = subprocess.run([sys.executable, "-m", "py_compile", "app.py"], cwd=ROOT, capture_output=True, text=True)
    chk("app.py compile", r.returncode == 0, r.stderr.strip()[-200:])
    try:
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        chk("app chạy không exception", not at.exception, str([e.value for e in at.exception])[:200])
    except Exception as e:
        chk("app chạy (AppTest)", False, f"{type(e).__name__}: {e}")


def phase5():
    exists("requirements.txt", "README.md", "tests/test_e2e.py")
    req = txt("requirements.txt").lower()
    chk("requirements đủ gói", all(k in req for k in ("pandas", "numpy", "scikit-learn", "torch", "matplotlib", "streamlit", "joblib", "openpyxl")))
    rd = txt("README.md")
    chk("README có lệnh chạy", all(k in rd for k in ("make_synthetic_data", "train", "evaluate", "streamlit run")))
    pytest("tests")


PHASES = {"1": phase1, "2": phase2, "3": phase3, "4": phase4, "5": phase5}

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k in (PHASES if arg == "all" else [arg]):
        try:
            PHASES[k]()
        except Exception as e:
            chk(f"phase{k} crash", False, f"{type(e).__name__}: {e}")
    bad = [r for r in R if not r[0]]
    for ok, n, d in R:
        print(("PASS " if ok else "FAIL ") + n + (f" | {d}" if d and not ok else ""))
    print(f"\nGATE {arg}: {'PASS' if not bad else 'FAIL'} ({len(R) - len(bad)}/{len(R)})")
    sys.exit(1 if bad else 0)
