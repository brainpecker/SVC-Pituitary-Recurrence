import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# =========================
# Page config
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
DEFAULT_BACKGROUND_CSV = "train.csv"

# =========================
# Title
# =========================
st.title("🧠 SVC Clinical Risk Prediction Dashboard")
st.caption(
    "Enter patient features to obtain prediction probability/class. "
    "SHAP explanation uses a built-in background dataset (train.csv)."
)

# =========================
# Utils
# =========================
def ensure_features(df, features):
    missing = [c for c in features if c not in df.columns]
    return len(missing) == 0, missing


def load_background_df(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        ok, _ = ensure_features(df, FEATURES)
        if not ok:
            return None
        return df[FEATURES].dropna().copy()
    except Exception:
        return None


# =========================
# Pure matplotlib SHAP force-style plot
# =========================
def plot_shap_force_style(
    shap_values,
    feature_values,
    feature_names,
    base_value,
    pred_value,
    title,
):
    shap_values = np.array(shap_values)
    order = np.argsort(np.abs(shap_values))[::-1]

    fig, ax = plt.subplots(figsize=(13, 3.6), dpi=200)

    pos = base_value
    y = 0

    for i in order:
        val = shap_values[i]
        name = feature_names[i]
        fval = feature_values[i]

        color = "#ff0051" if val > 0 else "#008bfb"

        ax.barh(
            y,
            val,
            left=pos,
            height=0.4,
            color=color,
            edgecolor="none",
        )

        ax.text(
            pos + val / 2,
            y - 0.38,
            f"{name} = {int(fval)}",
            ha="center",
            va="center",
            fontsize=9,
            color=color,
        )

        pos += val

    ax.axvline(base_value, color="gray", linestyle="--", linewidth=1)

    ax.set_yticks([])
    ax.set_xlabel("Predicted probability")

    ax.text(
        base_value,
        y + 0.45,
        f"baseline = {base_value:.2f}",
        ha="center",
        fontsize=10,
        color="gray",
    )
    ax.text(
        pred_value,
        y + 0.45,
        f"f(x) = {pred_value:.2f}",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )

    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    return fig


# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ Model")

model_path = st.sidebar.text_input("Model path", value=DEFAULT_MODEL_PATH)

try:
    model = joblib.load(model_path)
    st.sidebar.success("Model loaded successfully ✅")
except Exception as e:
    st.sidebar.error(f"Model load failed: {e}")
    st.stop()

show_explain = st.sidebar.checkbox("Show SHAP explanation", value=True)
plot_type = st.sidebar.radio(
    "SHAP plot type",
    ["Waterfall (paper)", "Force (paper-style bar)"],
    index=1,
)
bg_rows = st.sidebar.slider("SHAP background rows", 20, 500, 100, 10)
nsamples = st.sidebar.slider("SHAP nsamples", 50, 800, 200, 50)

# =========================
# Load SHAP background
# =========================
if "shap_bg" not in st.session_state:
    bg = load_background_df(DEFAULT_BACKGROUND_CSV)
    if bg is not None:
        st.session_state["shap_bg"] = bg
        st.sidebar.success(f"Background loaded ({len(bg)} rows)")
    else:
        st.sidebar.warning("No train.csv found for SHAP background")

# =========================
# Cache explainer
# =========================
@st.cache_resource
def get_shap_explainer(_predict_fn, bg_np):
    return shap.KernelExplainer(_predict_fn, bg_np)

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["🧍 Single Case", "📄 Batch (CSV)"])

# =========================
# Tab 1: Single case
# =========================
with tab1:
    st.subheader("Single Case Prediction")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        vi = st.selectbox("Visual impairment", [0, 1])
        ci = st.selectbox("Clival invasion", [0, 1])
    with c2:
        hardy = st.selectbox("Hardy D-E", [0, 1])
        p53 = st.selectbox("p53 positivity", [0, 1])
    with c3:
        ki67 = st.selectbox("Ki-67≥3%", [0, 1])
        subtype = st.selectbox("High-risk subtype", [0, 1])
    with c4:
        residual = st.selectbox("Residual tumor", [0, 1])

    X_one = pd.DataFrame([{
        "Visual impairment": vi,
        "Clival invasion": ci,
        "Hardy D-E": hardy,
        "p53 positivity": p53,
        "Ki-67≥3%": ki67,
        "High-risk subtype": subtype,
        "Residual tumor": residual,
    }])

    st.dataframe(X_one, use_container_width=True)

    thresh = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)

    if st.button("🔮 Predict"):
        proba = float(model.predict_proba(X_one)[:, 1][0])
        pred = int(proba >= thresh)

        st.metric("pred_proba", f"{proba:.4f}")
        st.success("Low risk (0)" if pred == 0 else "High risk (1)")

        if show_explain and "shap_bg" in st.session_state:
            bg = st.session_state["shap_bg"].head(bg_rows)
            bg_np = bg.to_numpy(float)
            x_np = X_one.to_numpy(float)

            def predict_fn(x):
                return model.predict_proba(pd.DataFrame(x, columns=FEATURES))

            explainer = get_shap_explainer(predict_fn, bg_np)
            shap_values = explainer.shap_values(x_np, nsamples=nsamples)
            expected_value = explainer.expected_value

            sv = shap_values[1] if isinstance(shap_values, list) else shap_values
            sv_arr = np.asarray(sv)

            if sv_arr.ndim == 3:
                sv_1d = sv_arr[0, :, 1]
            elif sv_arr.ndim == 2:
                sv_1d = sv_arr[0]
            else:
                sv_1d = sv_arr

            ev = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value

            st.markdown("### Model explanation (SHAP)")

            if plot_type.startswith("Force"):
                title = ("High-risk" if pred == 1 else "Low-risk") + \
                        f" patient  f(x) = {proba:.2f}, baseline = {ev:.2f}"

                fig = plot_shap_force_style(
                    sv_1d, x_np[0], FEATURES, ev, proba, title
                )
                st.pyplot(fig, use_container_width=True)

                png = io.BytesIO()
                fig.savefig(png, dpi=300, bbox_inches="tight")
                png.seek(0)
                st.download_button("⬇️ Download PNG", png, "shap_force.png")

                pdf = io.BytesIO()
                fig.savefig(pdf, format="pdf", bbox_inches="tight")
                pdf.seek(0)
                st.download_button("⬇️ Download PDF", pdf, "shap_force.pdf")

                plt.close(fig)

            else:
                exp = shap.Explanation(
                    values=sv_1d,
                    base_values=ev,
                    data=x_np[0],
                    feature_names=FEATURES,
                )
                fig = plt.figure(figsize=(10, 4.8), dpi=200)
                shap.plots.waterfall(exp, show=False)
                st.pyplot(fig)
                plt.close(fig)

# =========================
# Tab 2: Batch
# =========================
with tab2:
    st.subheader("Batch Prediction")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        ok, missing = ensure_features(df, FEATURES)
        if not ok:
            st.error(f"Missing columns: {missing}")
            st.stop()

        df["pred_proba"] = model.predict_proba(df[FEATURES])[:, 1]
        df["pred_label"] = model.predict(df[FEATURES])
        st.dataframe(df.head(20), use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Download results", csv, "svc_predictions.csv")
