"""Kiểm thử giao diện Streamlit bằng AppTest (không mở trình duyệt). Chạy vài giây."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NEEDED = ["preprocessor.joblib", "ridge.joblib", "rf.joblib", "mlp.pt"]

if not all((ROOT / "models" / f).exists() for f in NEEDED) or not (ROOT / "data" / "test.csv").exists():
    pytest.skip("Thiếu models/ hoặc data/test.csv (chạy python -m src.train)", allow_module_level=True)
pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

MODELS = ["Ridge", "Random Forest", "MLP (PyTorch)"]


@pytest.fixture()
def app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not at.exception
    return at


def _downloads(at):
    return len(at.get("download_button"))


def test_start_shows_synthetic_banner_and_no_result(app):
    assert any("SYNTHETIC" in m.value for m in app.markdown)
    assert _downloads(app) == 0


@pytest.mark.parametrize("model", MODELS)
def test_single_model_run(app, model):
    app.sidebar.radio[0].set_value(model).run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert _downloads(app) == 2  # CSV + XLSX


def test_compare_run_has_three_models_and_comparison_tab(app):
    app.sidebar.radio[0].set_value("So sánh cả 3").run()
    app.button[0].click().run()
    assert not app.exception and not app.error
    assert _downloads(app) == 6
    assert "So sánh" in [t.label for t in app.tabs]


def test_changing_campaign_clears_old_result(app):
    app.button[0].click().run()
    assert _downloads(app) == 2
    sel = app.sidebar.selectbox[0]
    sel.set_value(sel.options[1]).run()
    assert not app.exception
    assert _downloads(app) == 0


def test_slider_keeps_result_and_writes_no_files(app):
    out = ROOT / "outputs"
    before = len(list(out.glob("ranking_*"))) if out.exists() else 0
    app.button[0].click().run()
    app.sidebar.slider[0].set_value(5).run()
    assert not app.exception
    assert _downloads(app) == 2
    after = len(list(out.glob("ranking_*"))) if out.exists() else 0
    assert after == before  # xuất file trong bộ nhớ, không ghi vào outputs/
