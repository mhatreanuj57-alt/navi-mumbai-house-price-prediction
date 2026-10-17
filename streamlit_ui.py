import os
import sys
from pathlib import Path

# Limit numerical workers before importing NumPy/scikit-learn on small hosts.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Existing Render services use `python app.py`; start the Streamlit server too.
if __name__ == "__main__" and get_script_run_ctx(suppress_warning=True) is None:
    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", str(Path(__file__).resolve()),
        "--server.address=0.0.0.0",
        "--server.port=" + os.environ.get("PORT", "8501"),
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    raise SystemExit(stcli.main())

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "nm_model.joblib"
BMC_MODEL_PATH = ROOT / "models" / "bmc_model.joblib"

NODE_COORDS = {
    "Kharghar": (19.0968, 73.0741), "Ulwe": (18.9636, 73.0575),
    "Kamothe": (19.1166, 73.0892), "Panvel": (18.9894, 73.1175),
    "Taloja": (19.0797, 73.1213), "Seawoods": (19.0174, 73.0228),
    "Belapur": (19.0059, 73.0329), "Ghansoli": (19.1176, 73.0063),
    "Sanpada": (19.0645, 73.0094), "Airoli": (19.1556, 72.9987),
    "Dronagiri": (18.9044, 72.9976), "Nerul": (19.0348, 73.0197),
    "Vashi": (19.0771, 72.9986),
}

CONFIGS = {
    "1RK":  {"bhk": 1, "sqft": 300},
    "1BHK": {"bhk": 1, "sqft": 450},
    "2BHK": {"bhk": 2, "sqft": 720},
    "3BHK": {"bhk": 3, "sqft": 1075},
    "4BHK": {"bhk": 4, "sqft": 1520},
}

st.set_page_config(page_title="Navi Mumbai / BMC House Price Finder", page_icon="🏠", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-padding { padding-top: 1rem !important; }

.hero {
    background: linear-gradient(135deg, #064E3B 0%, #059669 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 22px;
}
.hero h1 { color: #ffffff; font-size: 1.9rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.hero p { color: #A7F3D0; font-size: 0.95rem; margin: 6px 0 0 0; }

.panel {
    background: #ffffff;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.panel-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #6B7280; }

.price-card {
    background: linear-gradient(135deg, #064E3B 0%, #059669 100%);
    border-radius: 16px;
    padding: 30px 32px;
    text-align: center;
}
.price-card .label { color: #A7F3D0; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; }
.price-card .amount { color: #ffffff; font-size: 3rem; font-weight: 800; line-height: 1.15; letter-spacing: -1px; }
.price-card .meta { color: #D1FAE5; font-size: 0.9rem; margin-top: 4px; }

.chip-row { display: flex; gap: 10px; justify-content: center; margin-top: 14px; flex-wrap: wrap; }
.chip { background: rgba(255,255,255,0.12); border-radius: 999px; padding: 6px 16px; color: #ECFDF5; font-size: 0.82rem; font-weight: 600; }

.ladder-card {
    background: #ffffff; border: 1px solid #E5E7EB; border-radius: 12px;
    padding: 18px 14px; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.ladder-card.active { border: 2px solid #059669; background: #ECFDF5; }
.ladder-card .cfg { font-size: 0.78rem; font-weight: 700; color: #6B7280; text-transform: uppercase; letter-spacing: 1px; }
.ladder-card .amt { font-size: 1.35rem; font-weight: 800; color: #064E3B; margin: 6px 0 2px; }
.ladder-card .rate { font-size: 0.78rem; color: #9CA3AF; }

.metric-tile {
    background: #ffffff; border: 1px solid #E5E7EB; border-radius: 12px;
    padding: 16px 20px; height: 100%;
}
.metric-tile .v { font-size: 1.45rem; font-weight: 800; color: #064E3B; }
.metric-tile .k { font-size: 0.78rem; font-weight: 600; color: #6B7280; text-transform: uppercase; letter-spacing: 0.8px; }
.metric-tile .s { font-size: 0.75rem; color: #9CA3AF; margin-top: 2px; }

[data-testid="stDataFrame"] { border: 1px solid #E5E7EB; border-radius: 12px; overflow: hidden; }
div[data-baseweb="select"] > div { border-radius: 10px; }
.stRadio div[role="radiogroup"] { gap: 0.4rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_bmc_bundle():
    return joblib.load(BMC_MODEL_PATH)


def fmt_inr(p):
    if p >= 1_00_00_000:
        return f"₹{p / 1_00_00_000:.2f} Cr"
    return f"₹{p / 1_00_000:.1f} L"


bundle = load_bundle()
model = bundle["model"]

df = pd.read_csv(ROOT / "data" / "navimumbai_listings.csv")
node_medians = df.groupby("node")["amenities"].median()
bmc_df = pd.read_csv(ROOT / "data" / "mumbai_bmc_listings.csv")

tab_finder, tab_insights = st.tabs(["🏠 Price Finder", "🔍 Model Insights"])

# ----------------------------- PRICE FINDER ---------------------------------
with tab_finder:
    st.markdown(
        '<div class="hero"><h1>🏠 Navi Mumbai / BMC House Price Finder</h1>'
        '<p>Choose Navi Mumbai or Mumbai / BMC · locality-level ML price estimates</p></div>',
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns([1, 1.25], gap="large")

    with c_left:
        region = st.radio("Select property region", ["Navi Mumbai", "Mumbai / BMC"], horizontal=True, key="region_select")
        query = st.text_input("🔎 Search locality", placeholder="Type a locality name...")
        nodes = sorted(NODE_COORDS.keys()) if region == "Navi Mumbai" else sorted(bmc_df["node"].unique())
        options = [n for n in nodes if query.lower() in n.lower()] or nodes
        default_node = "Kharghar" if region == "Navi Mumbai" else "Andheri"
        node = st.selectbox("Locality", options,
                            index=options.index(default_node) if default_node in options else 0,
                            key="locality_select")

        config = st.radio("Configuration", list(CONFIGS.keys()),
                          horizontal=True, key=f"config_radio")
        cfg = CONFIGS[config]

        sqft = st.slider("Carpet area (sq ft)", 200, 3000,
                         min(max(cfg["sqft"], 200), 3000), step=10,
                         key="area_slider_global")
        is_new = st.toggle("New property", value=True, key="new_toggle")

    with c_right:
        if region == "Navi Mumbai":
            amen_med = int(node_medians.get(node, 5))
            row = pd.DataFrame([{"sqft": sqft, "bhk": cfg["bhk"], "is_new": int(is_new), "amenities": amen_med, "node": node}])
            active_model, ward_label = model, ""
        else:
            local = bmc_df[bmc_df["node"] == node]
            ward = local["bmc_ward"].mode().iloc[0]
            resale_age = int(local.loc[local["is_new"] == 0, "age_years"].median()) if (local["is_new"] == 0).any() else 5
            row = pd.DataFrame([{"sqft": sqft, "bhk": cfg["bhk"], "is_new": int(is_new), "age_years": 0 if is_new else resale_age, "node": node, "bmc_ward": ward}])
            active_model, ward_label = load_bmc_bundle()["model"], f" · {ward}"
            amen_med = None
        price = float(active_model.predict(row)[0])
        rate = price / sqft

        cond = "New" if is_new else "Resale"
        st.markdown(
            f"""<div class="price-card">
                <div class="label">{config} · {node}{ward_label} · {cond}</div>
                <div class="amount">{fmt_inr(price)}</div>
                <div class="meta">estimated market value</div>
                <div class="chip-row">
                    <span class="chip">₹{rate:,.0f} / sq ft</span>
                    <span class="chip">{sqft} sq ft carpet</span>
                    <span class="chip">±₹16.8L typical error</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
        if region == "Navi Mumbai":
            lat, lon = NODE_COORDS[node]
        else:
            lat, lon = local[["latitude", "longitude"]].median()
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=11)

    st.markdown("")
    st.markdown(f'<div class="panel-title">Price ladder · every configuration in {node}</div>',
                unsafe_allow_html=True)
    st.markdown("")

    cols = st.columns(len(CONFIGS))
    active_cfg = cfg["bhk"]
    ladder_prices = {}
    for col, (name, c) in zip(cols, CONFIGS.items()):
        if region == "Navi Mumbai":
            r = pd.DataFrame([{"sqft": c["sqft"], "bhk": c["bhk"], "is_new": int(is_new), "amenities": amen_med, "node": node}])
        else:
            r = pd.DataFrame([{"sqft": c["sqft"], "bhk": c["bhk"], "is_new": int(is_new), "age_years": 0 if is_new else resale_age, "node": node, "bmc_ward": ward}])
        p = float(active_model.predict(r)[0])
        ladder_prices[name] = p
        cls = "ladder-card active" if c["bhk"] == active_cfg else "ladder-card"
        col.markdown(
            f"""<div class="{cls}">
                <div class="cfg">{name}</div>
                <div class="amt">{fmt_inr(p)}</div>
                <div class="rate">{c['sqft']} sq ft · ₹{p/c['sqft']:,.0f}/sqft</div>
            </div>""",
            unsafe_allow_html=True,
        )

    l1, l2 = st.columns([1.4, 1], gap="large")
    with l1:
        st.markdown("")
        st.markdown('<div class="panel-title">Live chart · updates with every filter</div>',
                    unsafe_allow_html=True)
        chart_df = pd.DataFrame(
            {"Price (₹ Lakh)": {k: v / 1e5 for k, v in ladder_prices.items()}}
        )
        st.bar_chart(chart_df, height=320, x_label="Configuration")
    with l2:
        st.markdown("")
        st.markdown('<div class="panel-title">Your selection vs typical</div>',
                    unsafe_allow_html=True)
        sel_typ = ladder_prices[config] * cfg["sqft"] / CONFIGS[config]["sqft"]
        delta = sel_typ - ladder_prices[config]
        pct = delta / ladder_prices[config] * 100 if ladder_prices[config] else 0
        st.metric(
            f"Your {config} at {sqft} sq ft",
            fmt_inr(sel_typ),
            delta=f"{pct:+.1f}% vs typical area",
        )

# ---------------------------- MODEL INSIGHTS --------------------------------
@st.cache_data(show_spinner="Running 5-fold cross-validation...")
def get_cv_results():
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    import os
    os.environ["OMP_NUM_THREADS"] = "1"

    from train_navimumbai import build_pipeline

    num_cols, cat_cols = bundle["num_cols"], bundle["cat_cols"]
    X = df[num_cols + cat_cols]
    y = df["price"]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    reg_pred = cross_val_predict(build_pipeline(), X, y, cv=cv, n_jobs=1)

    edges = y.quantile([0, 0.25, 0.5, 0.75, 1.0]).values
    band_labels = ["Budget", "Mid-range", "Premium", "Luxury"]
    band_codes = pd.cut(y, bins=edges, labels=band_labels,
                        include_lowest=True).cat.codes.values

    clf_pipe = Pipeline([
        ("pre", ColumnTransformer([
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ])),
        ("clf", HistGradientBoostingClassifier(random_state=42, learning_rate=0.08)),
    ])
    band_pred = cross_val_predict(clf_pipe, X, band_codes, cv=cv, n_jobs=1)

    return (y.values, reg_pred,
            np.array(band_labels)[band_codes], np.array(band_labels)[band_pred],
            band_labels, edges)


with tab_insights:
    if not st.toggle("Run model diagnostics", value=False):
        st.info("Enable model diagnostics to calculate cross-validation scores and charts.")
        st.stop()

    y_true, y_reg_pred, band_true, band_pred, band_labels, edges = get_cv_results()

    mae = np.mean(np.abs(y_true - y_reg_pred))
    r2 = 1 - np.sum((y_true - y_reg_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    acc = accuracy_score(band_true, band_pred)

    tiles = [
        (fmt_inr(mae), "MAE · out-of-fold", "typical miss per listing"),
        (f"{r2:.3f}", "R² · out-of-fold", "variance explained"),
        (f"{acc*100:.1f}%", "Band accuracy", "exact price bracket"),
        (f"{len(y_true):,}", "Listings evaluated", "5-fold cross-validation"),
    ]
    tcols = st.columns(4)
    for col, (v, k, s) in zip(tcols, tiles):
        col.markdown(
            f'<div class="metric-tile"><div class="v">{v}</div>'
            f'<div class="k">{k}</div><div class="s">{s}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.subheader("🎯 Confusion matrix — price bands")
    edge_txt = " · ".join(fmt_inr(e) for e in edges)
    st.caption(f"Bands by market quartiles: {edge_txt}. Diagonal = correct bracket guesses.")

    cm = confusion_matrix(band_true, band_pred, labels=band_labels)
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_pct, cmap="Greens", vmin=0, vmax=100)
    ax.set_xticks(range(len(band_labels)), band_labels, fontsize=11)
    ax.set_yticks(range(len(band_labels)), band_labels, fontsize=11)
    ax.set_xlabel("Predicted band", fontsize=11)
    ax.set_ylabel("Actual band", fontsize=11)
    for i in range(len(band_labels)):
        for j in range(len(band_labels)):
            color = "white" if cm_pct[i, j] > 55 else "#1a3d1a"
            ax.text(j, i, f"{cm[i, j]}\n{cm_pct[i, j]:.0f}%",
                    ha="center", va="center", color=color, fontsize=11)
    fig.colorbar(im, ax=ax, label="% of actual band")
    st.pyplot(fig, width="content")

    report = classification_report(band_true, band_pred, labels=band_labels,
                                   output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(report).T.loc[band_labels].round(3)
    st.dataframe(rep_df, width="stretch")

    sc1, sc2 = st.columns(2, gap="large")
    with sc1:
        st.subheader("📈 Predicted vs Actual")
        st.caption("Points hugging the dashed line = accurate predictions")
        fig2, ax2 = plt.subplots(figsize=(6, 4.8))
        ax2.scatter(y_true, y_reg_pred, s=12, alpha=0.35, color="#059669")
        lims = [min(y_true.min(), y_reg_pred.min()), max(y_true.max(), y_reg_pred.max())]
        ax2.plot(lims, lims, "--", color="#DC2626", lw=1.5)
        ax2.set_xlabel("Actual price")
        ax2.set_ylabel("Predicted price")
        ax2.yaxis.set_major_formatter(lambda v, _: f"₹{v/1e7:.1f}Cr")
        ax2.xaxis.set_major_formatter(lambda v, _: f"₹{v/1e7:.0f}Cr")
        ax2.grid(alpha=0.25)
        st.pyplot(fig2, width="stretch")

    with sc2:
        st.subheader("📉 Error distribution")
        st.caption("Centered on zero = unbiased model")
        residuals = y_true - y_reg_pred
        fig3, ax3 = plt.subplots(figsize=(6, 4.8))
        ax3.hist(residuals / 1e5, bins=40, color="#3D5AFE", alpha=0.85)
        ax3.axvline(0, color="#DC2626", ls="--", lw=1.5)
        ax3.set_xlabel("Error (₹ lakh)")
        ax3.set_ylabel("Listings")
        ax3.grid(alpha=0.25)
        st.pyplot(fig3, width="stretch")
