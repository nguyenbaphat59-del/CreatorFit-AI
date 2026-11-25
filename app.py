"""CreatorFit AI - giao diện Streamlit.

Chỉ suy luận: gọi predict_and_rank trên mô hình đã huấn luyện, không fit/huấn luyện gì ở đây.
Chạy: streamlit run app.py
"""
from __future__ import annotations

import html
from itertools import combinations
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.export import ranking_filename, to_csv_bytes, to_excel_bytes
from src.predict import predict_and_rank
from src.schema import (
    CAMPAIGN_CATEGORICAL,
    CAMPAIGN_NUMERIC,
    CREATOR_CATEGORICAL,
    CREATOR_NUMERIC,
)

st.set_page_config(page_title="CreatorFit AI", page_icon="🎯", layout="wide")

ROOT = Path(__file__).resolve().parent
TEST_DATA_PATH = ROOT / "data" / "test.csv"
METRICS_PATH = ROOT / "outputs" / "metrics.csv"
CREATOR_REQUIRED = ["creator_id"] + list(CREATOR_NUMERIC) + list(CREATOR_CATEGORICAL)
CAMPAIGN_COLUMNS = ["campaign_id", "campaign_name"] + list(CAMPAIGN_NUMERIC) + list(CAMPAIGN_CATEGORICAL)

MODEL_LABELS = {"ridge": "Ridge", "rf": "Random Forest", "mlp": "MLP (PyTorch)"}
LABEL_TO_KEY = {v: k for k, v in MODEL_LABELS.items()}
COMPARE = "So sánh cả 3"
HIGH_SCORE = 70
ACCENT = "#4F46E5"
ACCENT_SOFT = "#A5B4FC"

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1180px;}
section[data-testid="stSidebar"] {border-right: 1px solid #E2E8F0;}
section[data-testid="stSidebar"] .stButton > button {width: 100%; border-radius: 10px; font-weight: 600;}
.stDownloadButton > button {border-radius: 10px; font-weight: 500;}
.stTabs [data-baseweb="tab-list"] {gap: 4px;}
[data-testid="stMetric"] {background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 12px 16px;}
[data-testid="stMetricLabel"] p {color: #64748B; font-size: .82rem;}
.cf-hero h1 {font-size: 1.9rem; font-weight: 700; margin: 0; color: #0F172A; letter-spacing: -0.02em;}
.cf-hero p {color: #64748B; margin: .3rem 0 0 0; font-size: .97rem;}
.cf-eyebrow {font-size: .75rem; font-weight: 600; color: #4F46E5; text-transform: uppercase; letter-spacing: .08em;}
.cf-h2 {font-size: 1.25rem; font-weight: 650; color: #0F172A; margin: 1.1rem 0 .1rem 0;}
.cf-muted {color: #64748B; font-size: .88rem;}
.cf-chip {display: inline-block; padding: 3px 11px; margin: 6px 6px 0 0; border-radius: 999px;
          background: #EEF2FF; color: #3730A3; font-size: .8rem; font-weight: 500;}
.cf-card {border: 1px solid #E2E8F0; border-radius: 14px; padding: 16px 18px; background: #FFFFFF; min-height: 132px;}
.cf-card.top {border-color: #C7D2FE; background: linear-gradient(180deg, #EEF2FF 0%, #FFFFFF 75%);}
.cf-rank {font-size: .72rem; font-weight: 600; color: #4F46E5; text-transform: uppercase; letter-spacing: .08em;}
.cf-name {font-size: 1.05rem; font-weight: 600; color: #0F172A; margin-top: .2rem;}
.cf-sub {font-size: .8rem; color: #64748B;}
.cf-score {font-size: 2rem; font-weight: 700; color: #0F172A; line-height: 1.1; margin-top: .6rem;}
.cf-score span {font-size: .85rem; font-weight: 500; color: #94A3B8; margin-left: 2px;}
.cf-warn {border: 1px solid #FDE68A; background: #FFFBEB; color: #92400E; border-radius: 10px;
          padding: 8px 14px; font-size: .86rem; margin: .9rem 0 .2rem 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------- dữ liệu
@st.cache_data(show_spinner=False)
def load_test_data(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False, max_entries=64)
def compute_rankings(model_keys: tuple, campaign: dict, creators: pd.DataFrame) -> dict:
    """Chạy predict_and_rank cho từng mô hình (kết quả được cache theo đầu vào)."""
    return {key: predict_and_rank(key, campaign, creators) for key in model_keys}


def explain_error(exc: Exception) -> str:
    msg = str(exc)
    if isinstance(exc, FileNotFoundError):
        return f"Chưa có mô hình đã huấn luyện ({msg}). Hãy chạy: python -m src.train"
    if isinstance(exc, (AttributeError, ImportError)) or "version" in msg.lower():
        return ("Mô hình được lưu bằng phiên bản thư viện khác (ví dụ scikit-learn) với môi trường hiện tại. "
                "Hãy huấn luyện lại bằng: python -m src.train")
    if isinstance(exc, (ValueError, KeyError)):
        return f"Không thể dự đoán với dữ liệu hiện tại: {msg}"
    return f"Đã xảy ra lỗi khi dự đoán: {msg}"


def validate_creator_upload(uploaded: pd.DataFrame, selected_campaign: str):
    missing = [c for c in CREATOR_REQUIRED if c not in uploaded.columns]
    if missing:
        return None, "CSV thiếu cột Creator bắt buộc: " + ", ".join(missing)
    creators = uploaded.copy()
    if creators.empty:
        return None, "CSV không có dòng Creator nào."
    if "campaign_id" in creators.columns:
        ids = creators["campaign_id"].dropna().astype(str).unique().tolist()
        if len(ids) > 1:
            return None, "CSV chứa nhiều campaign_id; chỉ tải Creator của một Campaign."
        if ids and ids[0] != str(selected_campaign):
            return None, f"campaign_id trong CSV ({ids[0]}) không khớp Campaign đang chọn ({selected_campaign})."
    bad = []
    for col in CREATOR_NUMERIC:
        converted = pd.to_numeric(creators[col], errors="coerce")
        if converted.isna().any() and creators[col].notna().any():
            bad.append(col)
        creators[col] = converted
    if bad:
        return None, "Các cột số có giá trị không hợp lệ: " + ", ".join(bad)
    if creators["creator_id"].isna().any():
        return None, "Cột creator_id không được để trống."
    if creators.duplicated(subset=["creator_id"]).any():
        return None, "CSV có creator_id bị trùng."
    if "creator_name" not in creators.columns:
        creators["creator_name"] = creators["creator_id"].astype(str)
    return creators, None


# ------------------------------------------------------------------ hiển thị
def show_df(df: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except Exception:  # Streamlit cũ chưa hỗ trợ width="stretch"
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def show_chart(chart: alt.Chart) -> None:
    try:
        st.altair_chart(chart, width="stretch")
    except Exception:
        st.altair_chart(chart, use_container_width=True)


def enrich(rank_df: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    out = rank_df.copy()
    out["creator_id"] = out["creator_id"].astype(str)
    extra = [c for c in ("niche", "platform", "country", "followers", "engagement_rate") if c in info.columns]
    if extra:
        side = info[["creator_id"] + extra].copy()
        side["creator_id"] = side["creator_id"].astype(str)
        out = out.merge(side.drop_duplicates("creator_id"), on="creator_id", how="left")
    return out


def score_chart(df: pd.DataFrame, top_n: int) -> alt.Chart:
    top = df.head(top_n).copy()
    top["label"] = top["creator_name"].astype(str) + "  ·  " + top["creator_id"].astype(str)
    top["score"] = top["fit_score_pred"].round(1)
    base = alt.Chart(top).encode(
        y=alt.Y("label:N", sort=list(top["label"]), title=None, axis=alt.Axis(labelLimit=260, ticks=False, domain=False)),
        x=alt.X("score:Q", scale=alt.Scale(domain=[0, 100]), title="Fit Score dự đoán (0-100)",
                axis=alt.Axis(grid=True, gridColor="#E2E8F0", domain=False, tickCount=5)),
    )
    bars = base.mark_bar(cornerRadiusEnd=5, size=18).encode(
        color=alt.condition(alt.datum["rank"] == 1, alt.value(ACCENT), alt.value(ACCENT_SOFT)),
        tooltip=[alt.Tooltip("rank:Q", title="Hạng"), alt.Tooltip("creator_name:N", title="Creator"),
                 alt.Tooltip("score:Q", title="Fit Score", format=".1f")],
    )
    labels = base.mark_text(align="left", dx=5, color="#334155", fontSize=12).encode(text=alt.Text("score:Q", format=".1f"))
    return (bars + labels).properties(height=max(120, 30 * len(top))).configure_view(strokeWidth=0)


def podium(ranked: pd.DataFrame) -> None:
    top = ranked.head(3)
    cols = st.columns(3)
    for i, (_, row) in enumerate(top.iterrows()):
        meta = " · ".join(str(row[c]) for c in ("niche", "platform", "country") if c in row and pd.notna(row[c]))
        with cols[i]:
            st.markdown(
                f"""<div class="cf-card {'top' if i == 0 else ''}">
                <div class="cf-rank">Hạng {int(row['rank'])}</div>
                <div class="cf-name">{html.escape(str(row['creator_name']))}</div>
                <div class="cf-sub">{html.escape(str(row['creator_id']))}{' · ' + html.escape(meta) if meta else ''}</div>
                <div class="cf-score">{row['fit_score_pred']:.1f}<span>/ 100</span></div></div>""",
                unsafe_allow_html=True,
            )


def ranking_table(ranked: pd.DataFrame) -> None:
    order = [c for c in ("rank", "creator_name", "creator_id", "niche", "platform", "country",
                         "followers", "engagement_rate", "fit_score_pred") if c in ranked.columns]
    config = {
        "rank": st.column_config.NumberColumn("Hạng", format="%d", width="small"),
        "creator_name": st.column_config.TextColumn("Creator"),
        "creator_id": st.column_config.TextColumn("ID", width="small"),
        "niche": st.column_config.TextColumn("Ngách"),
        "platform": st.column_config.TextColumn("Nền tảng"),
        "country": st.column_config.TextColumn("Quốc gia", width="small"),
        "followers": st.column_config.NumberColumn("Followers", format="%d"),
        "engagement_rate": st.column_config.NumberColumn("Engagement", format="%.3f"),
        "fit_score_pred": st.column_config.ProgressColumn("Fit Score", min_value=0, max_value=100, format="%.1f"),
    }
    show_df(ranked, column_order=order, column_config=config)


def model_view(key: str, ranked: pd.DataFrame, files: dict, top_n: int) -> None:
    scores = ranked["fit_score_pred"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Điểm cao nhất", f"{scores.max():.1f}")
    k2.metric("Điểm trung bình", f"{scores.mean():.1f}")
    k3.metric("Điểm thấp nhất", f"{scores.min():.1f}")
    k4.metric(f"Creator ≥ {HIGH_SCORE} điểm", f"{int((scores >= HIGH_SCORE).sum())} / {len(ranked)}")
    st.markdown('<div class="cf-h2">Top 3 phù hợp nhất</div>', unsafe_allow_html=True)
    podium(ranked)
    st.markdown('<div class="cf-h2">Biểu đồ điểm</div>', unsafe_allow_html=True)
    show_chart(score_chart(ranked, top_n))
    st.markdown('<div class="cf-h2">Bảng xếp hạng đầy đủ</div>', unsafe_allow_html=True)
    ranking_table(ranked)
    c1, c2, _ = st.columns([1, 1, 3])
    c1.download_button("Tải CSV", data=files["csv"], file_name=files["csv_name"], mime="text/csv",
                       key=f"dl_csv_{key}")
    c2.download_button("Tải Excel (xlsx)", data=files["xlsx"], file_name=files["xlsx_name"],
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key=f"dl_xlsx_{key}")


SHORT = {"ridge": "Ridge", "rf": "RF", "mlp": "MLP"}


def comparison_view(rankings: dict) -> None:
    merged = None
    for key, df in rankings.items():
        part = df[["creator_id", "creator_name", "rank"]].rename(columns={"rank": f"Hạng {SHORT[key]}"})
        merged = part if merged is None else merged.merge(part, on=["creator_id", "creator_name"])
    rank_cols = [c for c in merged.columns if c.startswith("Hạng ")]
    merged["Hạng TB"] = merged[rank_cols].mean(axis=1).round(1)
    merged["Lệch hạng"] = merged[rank_cols].max(axis=1) - merged[rank_cols].min(axis=1)
    merged = merged.sort_values(["Hạng TB", "creator_id"]).reset_index(drop=True)
    merged.insert(0, "Creator", merged["creator_name"].astype(str) + " (" + merged["creator_id"].astype(str) + ")")

    corr = merged[rank_cols].corr(method="spearman")
    pairs = [corr.loc[a, b] for a, b in combinations(rank_cols, 2)]
    consensus = sum(pairs) / len(pairs) if pairs else float("nan")
    tops = {MODEL_LABELS[k]: str(df.iloc[0]["creator_name"]) for k, df in rankings.items()}
    c1, c2 = st.columns([1, 2])
    c1.metric("Đồng thuận thứ hạng (Spearman TB)", f"{consensus:.3f}")
    with c2:
        st.markdown('<div class="cf-muted" style="margin-top:.4rem">Top 1 theo từng mô hình</div>', unsafe_allow_html=True)
        st.markdown("".join(f'<span class="cf-chip">{html.escape(m)}: {html.escape(n)}</span>' for m, n in tops.items()),
                    unsafe_allow_html=True)
    st.caption("Hạng TB càng thấp càng được các mô hình đồng thuận xếp cao; Lệch hạng = hạng thấp nhất - hạng cao nhất giữa các mô hình.")
    show_df(merged, column_order=["Creator"] + rank_cols + ["Hạng TB", "Lệch hạng"])


def metrics_tab() -> None:
    st.markdown('<div class="cf-h2">Kết quả đánh giá trên tập kiểm thử</div>', unsafe_allow_html=True)
    if not METRICS_PATH.exists():
        st.info("Chưa có outputs/metrics.csv. Hãy chạy: python -m src.evaluate")
        return
    try:
        m = pd.read_csv(METRICS_PATH)
    except Exception as exc:
        st.error(f"Không đọc được outputs/metrics.csv: {exc}")
        return
    if m.empty:
        st.info("outputs/metrics.csv hiện không có dòng dữ liệu.")
        return
    name = {"ridge": "Ridge", "rf": "Random Forest", "mlp": "MLP"}
    best = [("Sai số thấp nhất (RMSE)", "RMSE", True, "{:.2f}"),
            ("Thứ hạng tốt nhất (Spearman)", "spearman_mean", False, "{:.3f}"),
            ("NDCG@10 tốt nhất", "ndcg@10_mean", False, "{:.3f}")]
    cols = st.columns(len(best))
    for col, (label, field, low, fmt) in zip(cols, best):
        if field in m.columns and "model" in m.columns:
            row = m.loc[m[field].idxmin() if low else m[field].idxmax()]
            col.metric(label, name.get(str(row["model"]), str(row["model"])))
            col.caption(f"{field} = {fmt.format(row[field])}")
    show_df(m, column_config={
        "model": st.column_config.TextColumn("Mô hình", width="small"),
        "spearman_mean": st.column_config.NumberColumn("Spearman TB", format="%.3f"),
        "spearman_std": st.column_config.NumberColumn("Spearman độ lệch", format="%.3f"),
        "ndcg@10_mean": st.column_config.NumberColumn("NDCG@10 TB", format="%.3f"),
        "p@10_mean": st.column_config.NumberColumn("Precision@10 TB", format="%.2f"),
        "n_campaigns": st.column_config.NumberColumn("Số Campaign", format="%d"),
        "MAE": st.column_config.NumberColumn("MAE", format="%.2f"),
        "RMSE": st.column_config.NumberColumn("RMSE", format="%.2f"),
        "R2": st.column_config.NumberColumn("R²", format="%.3f"),
    })
    st.caption("Số liệu đọc trực tiếp từ outputs/metrics.csv do src.evaluate tạo; app không tự tính metrics. "
               "Nếu dữ liệu là synthetic, kết quả chỉ xác nhận pipeline chạy đúng, không phản ánh hiệu năng thực.")
    imgs = [ROOT / "outputs" / n for n in ("pred_vs_true.png", "metric_compare.png") if (ROOT / "outputs" / n).exists()]
    if imgs:
        with st.expander("Biểu đồ đánh giá"):
            for p in imgs:
                st.image(str(p))


# ------------------------------------------------------------------- khởi tạo
st.markdown(
    '<div class="cf-hero"><div class="cf-eyebrow">Creator ↔ Campaign matching</div>'
    '<h1>CreatorFit AI</h1><p>Chọn một Campaign, chấm Fit Score (0-100) và xếp hạng Creator phù hợp nhất.</p></div>',
    unsafe_allow_html=True,
)

if not TEST_DATA_PATH.exists():
    st.error("Không tìm thấy data/test.csv. Hãy đặt file vào thư mục data/ trước khi chạy app.")
    st.stop()
try:
    data = load_test_data(str(TEST_DATA_PATH), TEST_DATA_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Không đọc được data/test.csv: {exc}")
    st.stop()
if "campaign_id" not in data.columns or data["campaign_id"].dropna().empty:
    st.error("data/test.csv thiếu cột campaign_id hợp lệ.")
    st.stop()

campaign_ids = data["campaign_id"].dropna().astype(str).drop_duplicates().tolist()
names = {}
if "campaign_name" in data.columns:
    names = dict(zip(data.drop_duplicates("campaign_id")["campaign_id"].astype(str),
                     data.drop_duplicates("campaign_id")["campaign_name"].astype(str)))

with st.sidebar:
    st.markdown('<div class="cf-eyebrow">CreatorFit AI</div><div class="cf-h2" style="margin-top:.1rem">Thiết lập</div>',
                unsafe_allow_html=True)
    selected_campaign_id = st.selectbox("Campaign", campaign_ids, help="Mỗi lần chỉ chấm điểm cho một Campaign.",
                                        format_func=lambda c: f"{c} · {names[c]}" if c in names else c)
    model_choice = st.radio("Mô hình", list(MODEL_LABELS.values()) + [COMPARE])
    with st.expander("Dùng danh sách Creator riêng (CSV)"):
        uploaded_file = st.file_uploader("CSV Creator", type=["csv"], label_visibility="collapsed")
        st.caption("Cột bắt buộc: " + ", ".join(CREATOR_REQUIRED))
    n_in = int((data["campaign_id"].astype(str) == selected_campaign_id).sum())
    top_n = st.slider("Top-N trên biểu đồ", 3, min(30, n_in), min(10, n_in)) if n_in >= 4 else n_in
    run_clicked = st.button("Dự đoán & Xếp hạng", type="primary", key="run_ranking")
    st.caption("Mô hình đã huấn luyện sẵn; ứng dụng chỉ suy luận.")

campaign_rows = data[data["campaign_id"].astype(str) == selected_campaign_id].copy()
campaign = campaign_rows.iloc[0].copy()

creators, upload_error = campaign_rows, None
if uploaded_file is not None:
    try:
        creators, upload_error = validate_creator_upload(pd.read_csv(uploaded_file), selected_campaign_id)
    except Exception as exc:
        creators, upload_error = None, f"Không thể đọc CSV Creator: {exc}"
if upload_error:
    st.error(upload_error)
    st.info("Ứng dụng đang giữ danh sách Creator mặc định từ data/test.csv.")
    creators = campaign_rows
elif uploaded_file is not None:
    st.success(f"Đã nạp {len(creators)} Creator từ CSV.")
creators_info = creators.copy()  # giữ thuộc tính Creator để hiển thị
creators = creators.drop(columns=[c for c in CAMPAIGN_COLUMNS if c in creators.columns])

# Banner: dữ liệu synthetic
src_cols = [df["target_source"].astype(str).str.lower() for df in (campaign_rows, creators_info)
            if "target_source" in df.columns]
if any(s.eq("synthetic").any() for s in src_cols):
    st.markdown('<div class="cf-warn"><b>SYNTHETIC</b> · Dữ liệu và điểm chuẩn là dữ liệu tổng hợp. '
                'Kết quả chỉ xác nhận pipeline chạy đúng, không phản ánh hiệu năng thực.</div>', unsafe_allow_html=True)

tab_rank, tab_metrics = st.tabs(["Dự đoán & Xếp hạng", "Đánh giá mô hình"])

with tab_rank:
    cname = names.get(selected_campaign_id, "")
    st.markdown(f'<div class="cf-h2">Campaign {html.escape(selected_campaign_id)}'
                f'<span class="cf-muted"> {("· " + html.escape(cname)) if cname else ""}</span></div>',
                unsafe_allow_html=True)
    chips = [str(campaign[c]) for c in ("campaign_niche", "campaign_platform", "target_country", "goal") if c in campaign.index]
    st.markdown("".join(f'<span class="cf-chip">{html.escape(c)}</span>' for c in chips), unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Ngân sách", f"${float(campaign['budget_usd']):,.0f}")
    m2.metric("Followers tối thiểu", f"{float(campaign['min_followers']):,.0f}")
    m3.metric("Thời lượng", f"{int(campaign['duration_days'])} ngày")
    m4.metric("Creator đầu vào", f"{len(creators)}")

    signature = (selected_campaign_id, uploaded_file.name if (uploaded_file is not None and not upload_error) else None)
    if run_clicked:
        keys = tuple(MODEL_LABELS) if model_choice == COMPARE else (LABEL_TO_KEY[model_choice],)
        try:
            with st.spinner("Đang chấm điểm..."):
                rankings = compute_rankings(keys, campaign.to_dict(), creators)
            stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            files = {k: {
                "csv": to_csv_bytes(df), "xlsx": to_excel_bytes(df),
                "csv_name": ranking_filename(selected_campaign_id, k, "csv", stamp),
                "xlsx_name": ranking_filename(selected_campaign_id, k, "xlsx", stamp),
            } for k, df in rankings.items()}
            st.session_state["cf_result"] = {"signature": signature, "rankings": rankings,
                                             "files": files, "info": creators_info}
        except Exception as exc:
            st.session_state.pop("cf_result", None)
            st.error(explain_error(exc))

    result = st.session_state.get("cf_result")
    if result and result["signature"] == signature:
        top_n = min(top_n, len(creators))
        rankings = result["rankings"]
        views = {k: enrich(df, result["info"]) for k, df in rankings.items()}
        if len(rankings) == 1:
            key = next(iter(rankings))
            model_view(key, views[key], result["files"][key], top_n)
        else:
            tabs = st.tabs([MODEL_LABELS[k] for k in rankings] + ["So sánh"])
            for tab, key in zip(tabs, rankings):
                with tab:
                    model_view(key, views[key], result["files"][key], top_n)
            with tabs[-1]:
                comparison_view(rankings)
    elif not run_clicked:
        st.info("Chọn Campaign và mô hình ở thanh bên, rồi bấm **Dự đoán & Xếp hạng**.")

with tab_metrics:
    metrics_tab()
