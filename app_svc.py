import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

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
TARGET = "event"
DEFAULT_MODEL_PATH = "best_svc.pkl"

st.title("🧠 SVC Clinical Risk Prediction Dashboard")
st.caption(
    "Enter patient features or upload a CSV file to obtain SVC prediction probabilities and classes. "
    "For SHAP explanation, please upload a CSV once (used only as background data)."
)


def ensure_features(df: pd.DataFrame, features: list[str]):
    missing = [c for c in features if c not in df.columns]
    if missing:
        return False, f"Missing feature columns: {missing}"
    return True, ""


def safe_auc(y_true, proba):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, proba)


st.sidebar.header("⚙️ Model Loading")
model_path = st.sidebar.text_input("Model path (best_svc.pkl)", value=DEFAULT_MODEL_PATH)

try:
    model = joblib.load(model_path)
    st.sidebar.success("Model loaded successfully ✅")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

# Optional UI control
st.sidebar.markdown("---")
show_explain = st.sidebar.checkbox("Show SHAP explanation for single case", value=True)
bg_rows = st.sidebar.slider("SHAP background rows (from uploaded CSV)", 20, 300, 100, 10)

# Cache the explainer to avoid rebuilding repeatedly
@st.cache_resource
def get_shap_explainer(_model, bg: pd.DataFrame):
    # KernelExplainer works with any model that has predict_proba
    # Note: may be slow if background is large
    return shap.KernelExplainer(_model.predict_proba, bg)


tab1, tab2 = st.tabs(["🧍 Single Case Prediction", "📄 Batch Prediction (CSV)"])

with tab1:
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

    X_one = pd.DataFrame(
        [
            {
                "Visual impairment": vi,
                "Clival invasion": ci,
                "Hardy D-E": hardy,
                "p53 positivity": p53,
                "Ki-67≥3%": ki67,
                "High-risk subtype": subtype,
                "Residual tumor": residual,
            }
        ]
    )

    st.write("Input features:")
    st.dataframe(X_one, use_container_width=True)

    thresh = st.slider(
        "Threshold: predict High Risk (1) if pred_proba ≥ threshold",
        0.0,
        1.0,
        0.5,
        0.01,
    )

    if st.button("🔮 Predict"):
        proba = float(model.predict_proba(X_one)[:, 1][0])
        pred_by_thresh = int(proba >= thresh)

        m1, m2 = st.columns(2)
        m1.metric("pred_proba (positive class probability)", f"{proba:.4f}")
        m2.metric(f"pred_label (threshold {thresh:.2f})", f"{pred_by_thresh}")

        if pred_by_thresh == 1:
            st.error("Result: High risk (1)")
        else:
            st.success("Result: Low risk (0)")

        # ---- SHAP explanation plot (single case) ----
        if show_explain:
            st.markdown("### Model explanation (SHAP)")

            if "shap_bg" not in st.session_state or st.session_state["shap_bg"].empty:
                st.warning(
                    "No SHAP background data found.\n\n"
                    "Please go to the **Batch Prediction (CSV)** tab and upload a CSV once. "
                    "We will use the first rows as background data for explanation only."
                )
            else:
                bg = st.session_state["shap_bg"].copy()
                if len(bg) > bg_rows:
                    bg = bg.head(bg_rows)

                try:
                    explainer = get_shap_explainer(model, bg)

                    # shap_values: list for each class [0, 1]
                    shap_values = explainer.shap_values(X_one)

                    # Plot force plot for positive class (class 1)
                    plt.figure()
                    shap.force_plot(
                        explainer.expected_value[1],
                        shap_values[1],
                        X_one,
                        matplotlib=True,
                        show=False,
                    )
                    fig = plt.gcf()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                except Exception as e:
                    st.error(f"Failed to generate SHAP plot: {e}")

with tab2:
    st.subheader("Upload CSV → Batch Predict → Download Results")
    st.info(f"The CSV file must contain the 7 feature columns: {FEATURES}")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        df_in = pd.read_csv(uploaded)

        ok, msg = ensure_features(df_in, FEATURES)
        if not ok:
            st.error(msg)
            st.stop()

        # Save background for SHAP (used in tab1)
        bg = df_in[FEATURES].dropna().head(max(bg_rows, 50)).copy()
        if len(bg) >= 20:
            st.session_state["shap_bg"] = bg
            st.success(f"SHAP background data has been set ({len(bg)} rows).")
        else:
            st.warning(
                "Uploaded CSV has too few valid rows for SHAP background (need ≥ 20). "
                "Predictions will still work, but explanation may be unavailable."
            )

        X_batch = df_in[FEATURES].copy()
        proba = model.predict_proba(X_batch)[:, 1]
        pred = model.predict(X_batch)

        out = df_in.copy()
        out["pred_label"] = pred
        out["pred_proba"] = proba

        st.write("Prediction preview:")
        st.dataframe(out.head(20), use_container_width=True)

        csv_bytes = out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="⬇️ Download prediction CSV",
            data=csv_bytes,
            file_name="svc_predictions.csv",
            mime="text/csv",
        )
