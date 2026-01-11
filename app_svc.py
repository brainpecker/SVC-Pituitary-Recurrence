import os
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

# ======================================================
# Page config
# ======================================================
st.set_page_config(page_title="SVC Risk Predictor", layout="wide")

# ======================================================
# Constants
# ======================================================
FEATURES = [
    "Visual impairment",
    "Clival invasion",
    "Hardy D-E",
    "p53 positivity",
    "Ki-67≥3%",
    "High-risk subtype",
    "Residual tumor",
]

MODEL_PATH = "best_svc.pkl"
FIXED_BACKGROUND_CSV = "background_fixed_50.csv"

BG_RANDOM_STATE = 0
NSAMPLES = 200

# ======================================================
# Typography (journal style)
# ======================================================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["font.size"] = 8
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7

st.title("🧠 SVC Clinical Risk Prediction Dashboard")
st.caption(
    "Single-case prediction + notebook-matched baseline "
    "(fixed background) + journal-style SHAP force plot"
)

# ======================================================
# Helper functions
# ======================================================
def ensure_features(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

def scalar_expected_value(ev) -> float:
    return float(np.array(ev).ravel()[0])

FEATURE_LABEL_MAP = {
    "Visual impairment": "Visual impairment",
    "Clival invasion": "Clival invasion",
    "Hardy D-E": "Hardy grade D–E",
    "p53 positivity": "p53 positivity",
    "Ki-67≥3%": "Ki-67 ≥ 3%",
    "High-risk subtype": "High-risk subtype",
    "Residual tumor": "Residual tumor",
}

def clean_and_style_forceplot_texts(ax, fx, baseline):
    # remove internal texts
    for txt in ax.texts:
        t = txt.get_text().lower()
        if "f(x)" in t or "base value" in t:
            txt.set_visible(False)

    # move higher/lower
    for txt in ax.texts:
        if "higher" in txt.get_text() or "lower" in txt.get_text():
            x, y = txt.get_position()
            txt.set_position((x, y + 0.08))

    # replace biggest numeric with fx
    biggest_txt = None
    biggest_size = 0
    for txt in ax.texts:
        if re.fullmatch(r"-?\d+(\.\d+)?", txt.get_text().strip()):
            if txt.get_fontsize() > biggest_size:
                biggest_size = txt.get_fontsize()
                biggest_txt = txt
    if biggest_txt is not None:
        biggest_txt.set_text(f"{fx:.2f}")

    # rename labels and 1.0 -> 1
    for txt in ax.texts:
        if " = " in txt.get_text():
            left, right = txt.get_text().split(" = ", 1)
            left = FEATURE_LABEL_MAP.get(left.strip(), left.strip())
            right = re.sub(r"(-?\d+)\.0\b", r"\1", right)
            txt.set_text(f"{left} = {right}")

    ax.set_title(
        f"f(x) = {fx:.2f}, baseline = {baseline:.2f}",
        fontsize=10,
        pad=8,
    )

    for txt in ax.texts:
        txt.set_fontsize(7)

def plot_force_plot(base_value, shap_values, x_row, fx, baseline):
    shap.force_plot(
        base_value=base_value,
        shap_values=shap_values,
        features=x_row,
        feature_names=x_row.index.tolist(),
        matplotlib=True,
        show=False,
    )

    fig = plt.gcf()
    fig.set_size_inches(12.5, 2.35)
    fig.set_dpi(150)

    ax = plt.gca()
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("Predicted probability", fontsize=8)

    clean_and_style_forceplot_texts(ax, fx, baseline)
    plt.tight_layout(pad=1.1)

    return fig

# ======================================================
# Load model
# ======================================================
st.sidebar.header("⚙️ Model")

try:
    model = joblib.load(MODEL_PATH)
    st.sidebar.success("Model loaded successfully")
except Exception as e:
    st.sidebar.error(str(e))
    st.stop()

# ======================================================
# Load fixed background
# ======================================================
if not os.path.exists(FIXED_BACKGROUND_CSV):
    st.error(f"Missing {FIXED_BACKGROUND_CSV}")
    st.stop()

bg_df = pd.read_csv(FIXED_BACKGROUND_CSV)
ensure_features(bg_df)
bg_df = bg_df[FEATURES].dropna()

@st.cache_resource
def build_explainer(bg_np):
    def f_prob(x):
        x_df = pd.DataFrame(x, columns=FEATURES)
        return model.predict_proba(x_df)[:, 1]

    return shap.KernelExplainer(f_prob, bg_np)

explainer = build_explainer(bg_df.to_numpy(float))
base_value = scalar_expected_value(explainer.expected_value)
baseline = base_value

# ======================================================
# Single case input
# ======================================================
st.subheader("Single Case Input → Risk Prediction")

c1, c2, c3, c4 = st.columns(4)

with c1:
    vi = st.selectbox("Visual impairment", [0, 1], 0)
    ci = st.selectbox("Clival invasion", [0, 1], 0)

with c2:
    hardy = st.selectbox("Hardy D-E", [0, 1], 0)
    p53 = st.selectbox("p53 positivity", [0, 1], 0)

with c3:
    ki67 = st.selectbox("Ki-67≥3%", [0, 1], 0)
    subtype = st.selectbox("High-risk subtype", [0, 1], 0)

with c4:
    residual = st.selectbox("Residual tumor", [0, 1], 0)

X_one = pd.DataFrame([{
    "Visual impairment": vi,
    "Clival invasion": ci,
    "Hardy D-E": hardy,
    "p53 positivity": p53,
    "Ki-67≥3%": ki67,
    "High-risk subtype": subtype,
    "Residual tumor": residual,
}])

# ======================================================
# Prediction + SHAP
# ======================================================
if st.button("🔮 Predict"):
    proba = float(model.predict_proba(X_one)[:, 1][0])

    # ✅ 保留两位小数
    st.metric(
        "pred_proba (positive class probability)",
        f"{proba:.2f}"
    )

    np.random.seed(BG_RANDOM_STATE)
    sv = explainer.shap_values(X_one.to_numpy(float), nsamples=NSAMPLES)

    if isinstance(sv, list):
        sv = np.asarray(sv[0])

    sv_1d = sv[0]

    fig = plot_force_plot(
        base_value=base_value,
        shap_values=sv_1d,
        x_row=X_one.iloc[0],
        fx=proba,
        baseline=baseline,
    )

    st.pyplot(fig, use_container_width=True, clear_figure=False)

    # ---------------- Export ----------------
    tiff_buf = io.BytesIO()
    fig.savefig(
        tiff_buf,
        format="tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.02,
    )
    tiff_buf.seek(0)

    st.download_button(
        "⬇️ Download SHAP force plot (TIFF, 600 dpi)",
        data=tiff_buf,
        file_name="shap_force_plot.tiff",
        mime="image/tiff",
    )

    plt.close(fig)
