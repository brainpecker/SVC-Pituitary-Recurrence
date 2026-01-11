import os
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

# =========================
# Streamlit page
# =========================
st.set_page_config(page_title="SVC Risk Predictor", layout="wide")

FEATURES = [
    "Visual impairment",
    "Clival invasion",
    "Hardy D-E",
    "p53 positivity",
    "Ki-67≥3%",
    "High-risk subtype",
    "Residual tumor",
]

DEFAULT_MODEL_PATH = "best_svc.pkl"
# ✅ 固定 background 文件（与你 notebook 对齐的关键）
FIXED_BACKGROUND_CSV = "background_fixed_50.csv"

# 可选：如果 fixed 文件不存在，fallback 用这个（一般不会用到）
FALLBACK_BACKGROUND_CSV = "train.csv"

BG_RANDOM_STATE = 0
DEFAULT_NSAMPLES = 200
DEFAULT_THRESH = 0.5

# =========================
# Journal-style typography (Times New Roman)
# =========================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["font.size"] = 8
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 7
plt.rcParams["ytick.labelsize"] = 7

st.title("🧠 SVC Clinical Risk Prediction Dashboard")
st.caption("Single-case prediction + notebook-matched baseline (fixed background) + journal-style SHAP force plot")

# =========================
# Helpers
# =========================
def ensure_features(df: pd.DataFrame, features: list[str]):
    missing = [c for c in features if c not in df.columns]
    if missing:
        return False, f"Missing feature columns: {missing}"
    return True, ""

def load_background_df_from_path(path: str) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        return None
    try:
        bg = pd.read_csv(path)
        ok, _ = ensure_features(bg, FEATURES)
        if not ok:
            return None
        return bg[FEATURES].dropna().copy()
    except Exception:
        return None

def load_background_df_from_upload(uploaded_file) -> pd.DataFrame | None:
    if uploaded_file is None:
        return None
    try:
        bg = pd.read_csv(uploaded_file)
        ok, _ = ensure_features(bg, FEATURES)
        if not ok:
            return None
        return bg[FEATURES].dropna().copy()
    except Exception:
        return None

def scalar_expected_value(ev) -> float:
    """Force expected_value to a single float (avoid mean ambiguity)."""
    return float(np.array(ev).ravel()[0])

# Paper-friendly feature labels (journal style)
FEATURE_LABEL_MAP = {
    "Visual impairment": "Visual impairment",
    "Clival invasion": "Clival invasion",
    "Hardy D-E": "Hardy grade D–E",
    "p53 positivity": "p53 positivity",
    "Ki-67≥3%": "Ki-67 ≥ 3%",
    "High-risk subtype": "High-risk subtype",
    "Residual tumor": "Residual tumor",
}

def format_feature_label(name: str) -> str:
    return FEATURE_LABEL_MAP.get(name, name)

def clean_and_style_forceplot_texts(
    ax: plt.Axes,
    fx: float,
    baseline: float,
    *,
    feature_label_map: dict[str, str],
    label_fontsize: int = 7,
    title_fontsize: int = 10,
):
    # remove internal f(x) / base value
    for txt in ax.texts:
        t = txt.get_text().lower()
        if "f(x)" in t or "base value" in t:
            txt.set_visible(False)

    # move higher/lower up a bit
    for txt in ax.texts:
        t = txt.get_text()
        if "higher" in t or "lower" in t:
            x, y = txt.get_position()
            txt.set_position((x, y + 0.08))

    # replace biggest numeric with real fx
    biggest_txt = None
    biggest_size = 0.0
    for txt in ax.texts:
        s = txt.get_text().strip()
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            fs = float(txt.get_fontsize())
            if fs > biggest_size:
                biggest_size = fs
                biggest_txt = txt
    if biggest_txt is not None:
        biggest_txt.set_text(f"{fx:.2f}")

    # 1.0 -> 1 and rename feature label
    for txt in ax.texts:
        t = txt.get_text()
        if " = " in t:
            left, right = t.split(" = ", 1)
            left_clean = left.strip()
            mapped = feature_label_map.get(left_clean, left_clean)
            right = re.sub(r"(-?\d+)\.0\b", r"\1", right)
            txt.set_text(f"{mapped} = {right}")

    # title
    ax.set_title(
        f"f(x) = {fx:.2f}, baseline = {baseline:.2f}",
        fontsize=title_fontsize,
        pad=8,
    )

    # uniform text font size
    for txt in ax.texts:
        txt.set_fontsize(label_fontsize)

def plot_force_prob_paper(
    *,
    base_value: float,
    shap_values_1d: np.ndarray,
    x_row: pd.Series,
    fx: float,
    baseline: float,
) -> plt.Figure:
    shap.force_plot(
        base_value=base_value,  # ✅ scalar base_value
        shap_values=shap_values_1d,
        features=x_row,
        feature_names=x_row.index.tolist(),
        matplotlib=True,
        show=False,
    )

    fig = plt.gcf()
    fig.set_size_inches(12.5, 2.35)

    # ✅ Streamlit display: avoid huge dpi to prevent blank rendering
    fig.set_dpi(150)

    ax = plt.gca()
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("Predicted probability", fontsize=8)

    clean_and_style_forceplot_texts(
        ax=ax,
        fx=fx,
        baseline=baseline,
        feature_label_map={k: format_feature_label(k) for k in FEATURES},
        label_fontsize=7,
        title_fontsize=10,
    )

    ax.tick_params(axis="x", labelsize=7)
    plt.tight_layout(pad=1.1)
    return fig

# =========================
# Sidebar: model + SHAP settings
# =========================
st.sidebar.header("⚙️ Model Loading")
model_path = st.sidebar.text_input("Model path", value=DEFAULT_MODEL_PATH)

try:
    model = joblib.load(model_path)
    st.sidebar.success("Model loaded successfully ✅")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

st.sidebar.markdown("---")
show_explain = st.sidebar.checkbox("Show SHAP explanation (journal-style force plot)", value=True)

st.sidebar.subheader("SHAP Background (Notebook-matched)")
st.sidebar.write(f"Default background: `{FIXED_BACKGROUND_CSV}` (fixed, no sampling)")

# 可选：允许上传替换 background（若你想对照不同背景）
uploaded_bg = st.sidebar.file_uploader(
    "Optional: upload another background CSV (override)",
    type=["csv"],
    help="If uploaded, this will override background_fixed_50.csv. Must contain the 7 feature columns.",
)

nsamples = st.sidebar.slider("SHAP nsamples", 50, 800, DEFAULT_NSAMPLES, 50)

# =========================
# Load background (fixed first)
# =========================
def get_background_df() -> pd.DataFrame | None:
    # 1) uploaded override
    bg_up = load_background_df_from_upload(uploaded_bg)
    if bg_up is not None and not bg_up.empty:
        return bg_up

    # 2) fixed background file (recommended)
    bg_fixed = load_background_df_from_path(FIXED_BACKGROUND_CSV)
    if bg_fixed is not None and not bg_fixed.empty:
        return bg_fixed

    # 3) fallback
    bg_fallback = load_background_df_from_path(FALLBACK_BACKGROUND_CSV)
    if bg_fallback is not None and not bg_fallback.empty:
        return bg_fallback

    return None

bg_df = get_background_df()
if bg_df is None or bg_df.empty:
    st.sidebar.error(
        f"Background not found/invalid. Please ensure `{FIXED_BACKGROUND_CSV}` exists and has the 7 feature columns."
    )
    st.stop()
else:
    st.sidebar.success(f"Background loaded ✅ ({len(bg_df)} rows after dropna)")

# =========================
# Cache KernelExplainer (keyed by bg bytes + nsamples seed)
# =========================
@st.cache_resource
def build_kernel_explainer(bg_np: np.ndarray, seed: int):
    # Notebook style: explain probability of positive class
    def f_prob(x):
        x_df = pd.DataFrame(x, columns=FEATURES)
        return model.predict_proba(x_df)[:, 1]
    _ = seed
    return shap.KernelExplainer(f_prob, bg_np)

bg_np = bg_df.to_numpy(dtype=float)
explainer = build_kernel_explainer(bg_np, BG_RANDOM_STATE)

base_value = scalar_expected_value(explainer.expected_value)
baseline = base_value  # show the same baseline that force_plot uses

# =========================
# UI: Single case
# =========================
st.subheader("Single Case Input → Risk Prediction")

c1, c2, c3, c4 = st.columns(4)
with c1:
    vi = st.selectbox("Visual impairment", [0, 1], index=0)
    ci = st.selectbox("Clival invasion", [0, 1], index=0)
with c2:
    hardy = st.selectbox("Hardy D-E", [0, 1], index=0)
    p53 = st.selectbox("p53 positivity", [0, 1], index=0)
with c3:
    ki67 = st.selectbox("Ki-67≥3%", [0, 1], index=0)
    subtype = st.selectbox("High-risk subtype", [0, 1], index=0)
with c4:
    residual = st.selectbox("Residual tumor", [0, 1], index=0)

X_one = pd.DataFrame([{
    "Visual impairment": vi,
    "Clival invasion": ci,
    "Hardy D-E": hardy,
    "p53 positivity": p53,
    "Ki-67≥3%": ki67,
    "High-risk subtype": subtype,
    "Residual tumor": residual,
}])

st.write("Input features:")
st.dataframe(X_one, use_container_width=True)

thresh = st.slider("Threshold", 0.0, 1.0, DEFAULT_THRESH, 0.01)

if st.button("🔮 Predict"):
    # prediction
    proba = float(model.predict_proba(X_one)[:, 1][0])
    pred_by_thresh = int(proba >= thresh)

    m1, m2 = st.columns(2)
    m1.metric("pred_proba (positive class probability)", f"{proba:.4f}")
    m2.metric(f"pred_label (threshold {thresh:.2f})", f"{pred_by_thresh}")

    if pred_by_thresh == 1:
        st.error("Result: High risk (1)")
    else:
        st.success("Result: Low risk (0)")

    # SHAP force plot
    if show_explain:
        st.markdown("### Model explanation (SHAP)")
        st.caption(f"baseline (expected_value) = {baseline:.6f}  |  background rows = {len(bg_df)}")

        try:
            # deterministic-ish for kernel sampling
            np.random.seed(BG_RANDOM_STATE)

            x_np = X_one.to_numpy(dtype=float)
            sv = explainer.shap_values(x_np, nsamples=nsamples)

            if isinstance(sv, list):
                sv_arr = np.asarray(sv[0])
            else:
                sv_arr = np.asarray(sv)

            sv_1d = sv_arr[0] if sv_arr.ndim == 2 else sv_arr

            fig = plot_force_prob_paper(
                base_value=base_value,
                shap_values_1d=sv_1d,
                x_row=X_one.iloc[0],
                fx=proba,
                baseline=baseline,
            )

            st.pyplot(fig, use_container_width=True, clear_figure=False)

            # ===== Export (high dpi) =====
            # TIFF 600 dpi
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

            # PNG 300 dpi
            png_buf = io.BytesIO()
            fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
            png_buf.seek(0)
            st.download_button(
                "⬇️ Download SHAP force plot (PNG, 300 dpi)",
                data=png_buf,
                file_name="shap_force_plot.png",
                mime="image/png",
            )

            # PDF
            pdf_buf = io.BytesIO()
            fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
            pdf_buf.seek(0)
            st.download_button(
                "⬇️ Download SHAP force plot (PDF)",
                data=pdf_buf,
                file_name="shap_force_plot.pdf",
                mime="application/pdf",
            )

            plt.close(fig)

        except Exception as e:
            st.error(f"Failed to generate journal-style SHAP force plot: {e}")
