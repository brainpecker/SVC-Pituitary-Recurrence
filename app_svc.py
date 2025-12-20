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

st.title("🧠 SVC Clinical Risk Prediction Dashboard")
st.caption(
    "Enter patient features to obtain prediction probability/class. "
    "SHAP explanation uses a built-in background dataset (train.csv)."
)

# =========================
# Utilities
# =========================
def ensure_features(df: pd.DataFrame, features: list[str]):
    missing = [c for c in features if c not in df.columns]
    if missing:
        return False, f"Missing feature columns: {missing}"
    return True, ""

def load_background_df(path: str):
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

# =========================
# Sidebar – model
# =========================
st.sidebar.header("⚙️ Model")
model_path = st.sidebar.text_input("Model path", value=DEFAULT_MODEL_PATH)

try:
    model = joblib.load(model_path)
    st.sidebar.success("Model loaded ✅")
except Exception as e:
    st.sidebar.error(f"Model load failed: {e}")
    st.stop()

show_explain = st.sidebar.checkbox("Show SHAP explanation", value=True)
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
        st.sidebar.warning("No background CSV found (train.csv)")

# =========================
# Cache explainer
# =========================
@st.cache_resource
def get_shap_explainer(predict_fn, bg_np):
    return shap.KernelExplainer(predict_fn, bg_np)

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["🧍 Single Case", "📄 Batch (CSV)"])

# =========================
# Tab 1 – Single case
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
        # -------- Prediction --------
        proba = float(model.predict_proba(X_one)[:, 1][0])
        pred = int(proba >= thresh)

        m1, m2 = st.columns(2)
        m1.metric("pred_proba", f"{proba:.4f}")
        m2.metric("pred_label", pred)

        st.success("Low risk (0)" if pred == 0 else "High risk (1)")

        # -------- SHAP --------
        if show_explain:
            st.markdown("### Model explanation (SHAP)")
            st.caption("Paper-ready SHAP waterfall (static)")

            if "shap_bg" not in st.session_state:
                st.warning("No SHAP background available.")
            else:
                bg_df = st.session_state["shap_bg"].head(bg_rows)
                bg_np = bg_df.to_numpy(float)
                x_np = X_one.to_numpy(float)

                def predict_fn(x):
                    x_df = pd.DataFrame(x, columns=FEATURES)
                    return np.asarray(model.predict_proba(x_df))

                try:
                    explainer = get_shap_explainer(predict_fn, bg_np)
                    shap_values = explainer.shap_values(x_np, nsamples=nsamples)
                    expected_value = explainer.expected_value

                    # ---- Robust extraction to 1D (handles (1,7,2)) ----
                    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
                    sv_arr = np.asarray(sv)

                    if sv_arr.ndim == 3:
                        sv_1d = sv_arr[0, :, 1]
                    elif sv_arr.ndim == 2 and sv_arr.shape[1] > 1:
                        sv_1d = sv_arr[:, 1]
                    elif sv_arr.ndim == 2:
                        sv_1d = sv_arr[0]
                    else:
                        sv_1d = sv_arr

                    base_val = (
                        expected_value[1]
                        if isinstance(expected_value, (list, np.ndarray))
                        else expected_value
                    )

                    exp = shap.Explanation(
                        values=sv_1d,
                        base_values=base_val,
                        data=x_np[0],
                        feature_names=FEATURES,
                    )

                    fig = plt.figure(figsize=(10, 4.8), dpi=200)
                    shap.plots.waterfall(exp, max_display=len(FEATURES), show=False)
                    st.pyplot(fig, use_container_width=True)

                    # ---- Export ----
                    png = io.BytesIO()
                    fig.savefig(png, dpi=300, bbox_inches="tight")
                    png.seek(0)
                    st.download_button("⬇️ Download PNG (300 dpi)", png, "shap_waterfall.png")

                    pdf = io.BytesIO()
                    fig.savefig(pdf, format="pdf", bbox_inches="tight")
                    pdf.seek(0)
                    st.download_button("⬇️ Download PDF", pdf, "shap_waterfall.pdf")

                    plt.close(fig)

                except Exception as e:
                    st.error(f"SHAP failed: {e}")

# =========================
# Tab 2 – Batch
# =========================
with tab2:
    st.subheader("Batch Prediction")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        ok, msg = ensure_features(df, FEATURES)
        if not ok:
            st.error(msg)
            st.stop()

        X = df[FEATURES]
        df["pred_proba"] = model.predict_proba(X)[:, 1]
        df["pred_label"] = model.predict(X)

        st.dataframe(df.head(20), use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Download results", csv, "svc_predictions.csv")
