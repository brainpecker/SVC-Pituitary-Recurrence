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
DEFAULT_BACKGROUND_CSV = "train.csv"  # put next to app_svc.py

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
st.caption("Single-case prediction + journal-style SHAP force plot (matplotlib)")

# =========================
# Helpers
# =========================
def ensure_features(df: pd.DataFrame, features: list[str]):
    missing = [c for c in features if c not in df.columns]
    if missing:
        return False, f"Missing feature columns: {missing}"
    return True, ""


def load_background_df(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        bg = pd.read_csv(path)
        ok, msg = ensure_features(bg, FEATURES)
        if not ok:
            return None
        return bg[FEATURES].dropna().copy()
    except Exception:
        return None


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
    title_prefix: str,
    *,
    feature_label_map: dict[str, str],
    label_fontsize: int = 7,
    title_fontsize: int = 10,
):
    """
    Make SHAP matplotlib force_plot look journal-ready:
    - Remove internal f(x)/base value texts
    - Move higher/lower upward
    - Replace biggest numeric with true fx
    - Convert 1.0 -> 1
    - Rename feature labels (left side of "name = value")
    - Set uniform smaller fontsize for all ax.texts
    """
    # 1) remove internal f(x) / base value
    for txt in ax.texts:
        t = txt.get_text().lower()
        if "f(x)" in t or "base value" in t:
            txt.set_visible(False)

    # 2) move higher/lower up a bit
    for txt in ax.texts:
        t = txt.get_text()
        if "higher" in t or "lower" in t:
            x, y = txt.get_position()
            txt.set_position((x, y + 0.08))

    # 3) replace biggest numeric text with real fx
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

    # 4) convert "1.0" -> "1" in labels and rename feature
    for txt in ax.texts:
        t = txt.get_text()

        # rename feature labels of pattern: "<feature> = <value>"
        if " = " in t:
            # split once
            left, right = t.split(" = ", 1)
            left_clean = left.strip()

            # find mapping by exact match (robust to small formatting)
            mapped = feature_label_map.get(left_clean, left_clean)

            # 1.0 -> 1
            right = re.sub(r"(-?\d+)\.0\b", r"\1", right)

            txt.set_text(f"{mapped} = {right}")

    # 5) set title (only one consistent title)
    full_title = f"{title_prefix}  f(x) = {fx:.2f}, baseline = {baseline:.2f}"
    ax.set_title(full_title, fontsize=title_fontsize, pad=8)

    # 6) uniform small text to avoid overlap
    for txt in ax.texts:
        txt.set_fontsize(label_fontsize)


def plot_force_prob_paper(
    *,
    explainer,
    shap_values_1d: np.ndarray,
    x_row: pd.Series,
    fx: float,
    baseline: float,
    title_prefix: str,
) -> plt.Figure:
    """
    Create journal-style force plot figure using SHAP's matplotlib=True output,
    then post-process text/labels to match paper style.
    """
    # SHAP force plot
    shap.force_plot(
        base_value=explainer.expected_value,
        shap_values=shap_values_1d,
        features=x_row,
        feature_names=x_row.index.tolist(),
        matplotlib=True,
        show=False,
    )

    fig = plt.gcf()
    # slightly taller than before to reduce label collision
    fig.set_size_inches(12.5, 2.35)
    fig.set_dpi(600)

    ax = plt.gca()
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("Predicted probability", fontsize=8)

    # Apply journal styling
    clean_and_style_forceplot_texts(
        ax=ax,
        fx=fx,
        baseline=baseline,
        title_prefix=title_prefix,
        feature_label_map={k: format_feature_label(k) for k in FEATURES},
        label_fontsize=7,
        title_fontsize=10,
    )

    ax.tick_params(axis="x", labelsize=7)

    # tighter layout (journal style)
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

bg_rows = st.sidebar.slider("SHAP background rows", 20, 500, 50, 10)   # paper-like default
nsamples = st.sidebar.slider("SHAP nsamples", 50, 800, 200, 50)

DEFAULT_THRESH = 0.5

# =========================
# Load SHAP background (train.csv)
# =========================
if "shap_bg" not in st.session_state:
    auto_bg = load_background_df(DEFAULT_BACKGROUND_CSV)
    if auto_bg is not None:
        st.session_state["shap_bg"] = auto_bg
        st.sidebar.success(f"SHAP background loaded from {DEFAULT_BACKGROUND_CSV} ({len(auto_bg)} rows).")
    else:
        st.sidebar.warning(
            f"No built-in background found: {DEFAULT_BACKGROUND_CSV}. "
            "Please add train.csv next to the app file."
        )

# =========================
# Cache KernelExplainer by bg only (avoid hashing functions)
# =========================
@st.cache_resource
def build_kernel_explainer(bg_np: np.ndarray):
    # probability-space explainer (positive class)
    def f_prob(x):
        x_df = pd.DataFrame(x, columns=FEATURES)
        return model.predict_proba(x_df)[:, 1]

    return shap.KernelExplainer(f_prob, bg_np)


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
    # ---- prediction ----
    proba = float(model.predict_proba(X_one)[:, 1][0])
    pred_by_thresh = int(proba >= thresh)

    m1, m2 = st.columns(2)
    m1.metric("pred_proba (positive class probability)", f"{proba:.4f}")
    m2.metric(f"pred_label (threshold {thresh:.2f})", f"{pred_by_thresh}")

    if pred_by_thresh == 1:
        st.error("Result: High risk (1)")
        title_prefix = "High-risk patient"
    else:
        st.success("Result: Low risk (0)")
        title_prefix = "Low-risk patient"

    # ---- SHAP force plot ----
    if show_explain:
        st.markdown("### Model explanation (SHAP)")

        if "shap_bg" not in st.session_state or st.session_state["shap_bg"].empty:
            st.warning(f"No SHAP background data. Please put `{DEFAULT_BACKGROUND_CSV}` next to the app.")
        else:
            bg_df = st.session_state["shap_bg"].copy()
            if len(bg_df) > bg_rows:
                bg_df = bg_df.head(bg_rows)

            bg_np = bg_df.to_numpy(dtype=float)

            try:
                explainer = build_kernel_explainer(bg_np)

                x_np = X_one.to_numpy(dtype=float)
                sv = explainer.shap_values(x_np, nsamples=nsamples)

                # Normalize to 1D (n_features,)
                if isinstance(sv, list):
                    sv_arr = np.asarray(sv[0])
                else:
                    sv_arr = np.asarray(sv)

                if sv_arr.ndim == 2:
                    sv_1d = sv_arr[0]
                else:
                    sv_1d = sv_arr

                # baseline like your original: mean of expected_value
                baseline = float(np.array(explainer.expected_value).mean())

                fig = plot_force_prob_paper(
                    explainer=explainer,
                    shap_values_1d=sv_1d,
                    x_row=X_one.iloc[0],
                    fx=proba,
                    baseline=baseline,
                    title_prefix=title_prefix,
                )

                st.pyplot(fig, use_container_width=True)

                # ---- Export TIFF (600 dpi) ----
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

                # ---- Export PNG (300 dpi) ----
                png_buf = io.BytesIO()
                fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
                png_buf.seek(0)
                st.download_button(
                    "⬇️ Download SHAP force plot (PNG, 300 dpi)",
                    data=png_buf,
                    file_name="shap_force_plot.png",
                    mime="image/png",
                )

                # ---- Export PDF ----
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
