import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

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

st.title("🧠 SVC Clinical Risk Prediction Dashboard")
st.caption(
    "Enter patient features to obtain prediction probability/class. "
    "SHAP explanation uses a built-in background dataset (train.csv), so users don't need to upload anything."
)


def ensure_features(df: pd.DataFrame, features: list[str]):
    missing = [c for c in features if c not in df.columns]
    if missing:
        return False, f"Missing feature columns: {missing}"
    return True, ""


def load_background_df(path: str) -> pd.DataFrame | None:
    """Load background data for SHAP from CSV."""
    if not os.path.exists(path):
        return None
    try:
        bg = pd.read_csv(path)
        ok, msg = ensure_features(bg, FEATURES)
        if not ok:
            st.sidebar.warning(f"Background CSV invalid: {msg}")
            return None
        bg = bg[FEATURES].dropna().copy()
        return bg
    except Exception as e:
        st.sidebar.warning(f"Failed to load background CSV: {e}")
        return None


# --- Sidebar: model loading ---
st.sidebar.header("⚙️ Model Loading")
model_path = st.sidebar.text_input("Model path (best_svc.pkl)", value=DEFAULT_MODEL_PATH)

try:
    model = joblib.load(model_path)
    st.sidebar.success("Model loaded successfully ✅")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

st.sidebar.markdown("---")
show_explain = st.sidebar.checkbox("Show SHAP explanation for single case", value=True)
plot_type = st.sidebar.radio(
    "SHAP plot type",
    ["Waterfall (paper)", "Force (paper-style bar)"],
    index=0,
)
bg_rows = st.sidebar.slider("SHAP background rows", 20, 500, 100, 10)
nsamples = st.sidebar.slider("SHAP nsamples (speed/quality)", 50, 800, 200, 50)

# --- Auto-load background from train.csv (so others don't need to upload) ---
if "shap_bg" not in st.session_state:
    auto_bg = load_background_df(DEFAULT_BACKGROUND_CSV)
    if auto_bg is not None:
        st.session_state["shap_bg"] = auto_bg
        st.sidebar.success(
            f"SHAP background loaded from {DEFAULT_BACKGROUND_CSV} ({len(auto_bg)} rows)."
        )
    else:
        st.sidebar.warning(
            f"No built-in background found: {DEFAULT_BACKGROUND_CSV}. "
            "Please add train.csv next to the app file (recommended), "
            "or upload a CSV in Batch tab to set background."
        )


# ✅ IMPORTANT FIX: leading underscore prevents Streamlit from hashing function objects
@st.cache_resource
def get_shap_explainer(_predict_fn, bg_np: np.ndarray):
    return shap.KernelExplainer(_predict_fn, bg_np)


tab1, tab2 = st.tabs(["🧍 Single Case Prediction", "📄 Batch Prediction (CSV)"])

# =========================
# Tab 1: Single case
# =========================
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

    thresh = st.slider(
        "Threshold: predict High Risk (1) if pred_proba ≥ threshold",
        0.0, 1.0, 0.5, 0.01
    )

    if st.button("🔮 Predict"):
        # ---- prediction ----
        proba = float(model.predict_proba(X_one)[:, 1][0])
        pred_by_thresh = int(proba >= thresh)

        m1, m2 = st.columns(2)
        m1.metric("pred_proba (positive class probability)", f"{proba:.4f}")
        m2.metric(f"pred_label (threshold {thresh:.2f})", f"{pred_by_thresh}")

        if pred_by_thresh == 1:
            st.error("Result: High risk (1)")
        else:
            st.success("Result: Low risk (0)")

        # ---- SHAP (Waterfall or Force) ----
        if show_explain:
            st.markdown("### Model explanation (SHAP)")
            st.caption("Paper-ready SHAP plot with PNG/PDF export.")

            if "shap_bg" not in st.session_state or st.session_state["shap_bg"].empty:
                st.warning(
                    "No SHAP background data available.\n\n"
                    f"Please add `{DEFAULT_BACKGROUND_CSV}` next to the app (recommended), "
                    "or upload a CSV in the Batch tab once."
                )
            else:
                bg_df = st.session_state["shap_bg"].copy()
                if len(bg_df) > bg_rows:
                    bg_df = bg_df.head(bg_rows)

                bg_np = bg_df[FEATURES].to_numpy(dtype=float)
                x_np = X_one[FEATURES].to_numpy(dtype=float)

                def predict_fn(x):
                    x_df = pd.DataFrame(x, columns=FEATURES)
                    proba_ = np.asarray(model.predict_proba(x_df))
                    if proba_.ndim == 1:
                        proba_ = proba_.reshape(-1, 1)
                    return proba_

                try:
                    explainer = get_shap_explainer(predict_fn, bg_np)
                    shap_values = explainer.shap_values(x_np, nsamples=nsamples)
                    expected_value = explainer.expected_value

                    # choose positive-class container if possible
                    sv = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values
                    sv_arr = np.asarray(sv)

                    # ---- Robust: make 1D SHAP values for positive class ----
                    # shapes: (1,7,2) -> [0,:,1]
                    #         (7,2)   -> [:,1]
                    #         (1,7)   -> [0]
                    #         (7,)    -> 그대로
                    if sv_arr.ndim == 3:
                        out_idx = 1 if sv_arr.shape[2] > 1 else 0
                        sv_1d = sv_arr[0, :, out_idx]
                    elif sv_arr.ndim == 2:
                        if sv_arr.shape[0] == 1:
                            sv_1d = sv_arr[0]
                        elif sv_arr.shape[1] > 1:
                            sv_1d = sv_arr[:, 1]
                        else:
                            sv_1d = sv_arr[:, 0]
                    else:
                        sv_1d = sv_arr

                    # base value: take positive class if available
                    ev_arr = np.atleast_1d(expected_value)
                    base_val = ev_arr[1] if ev_arr.size > 1 else float(expected_value)

                    # Build Explanation (for waterfall)
                    exp = shap.Explanation(
                        values=sv_1d,
                        base_values=base_val,
                        data=x_np[0],
                        feature_names=FEATURES
                    )

                    # =========================
                    # Waterfall (paper)
                    # =========================
                    if plot_type.startswith("Waterfall"):
                        fig = plt.figure(figsize=(10, 4.8), dpi=200)
                        shap.plots.waterfall(exp, max_display=len(FEATURES), show=False)
                        st.pyplot(fig, use_container_width=True)

                        png_buf = io.BytesIO()
                        fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
                        png_buf.seek(0)
                        st.download_button(
                            "⬇️ Download SHAP waterfall (PNG, 300 dpi)",
                            data=png_buf,
                            file_name="shap_waterfall.png",
                            mime="image/png",
                        )

                        pdf_buf = io.BytesIO()
                        fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")
                        pdf_buf.seek(0)
                        st.download_button(
                            "⬇️ Download SHAP waterfall (PDF)",
                            data=pdf_buf,
                            file_name="shap_waterfall.pdf",
                            mime="application/pdf",
                        )
                        plt.close(fig)

                    # =========================
                    # Force (paper-style bar)
                    # =========================
                    else:
                        # This produces the style similar to your screenshot
                        fig = plt.figure(figsize=(13, 3.2), dpi=200)
                        shap.force_plot(
                            base_value=base_val,
                            shap_values=sv_1d,
                            features=x_np[0],
                            feature_names=FEATURES,
                            matplotlib=True,
                            show=False,
                        )
                        st.pyplot(fig, use_container_width=True)

                        png_buf = io.BytesIO()
                        fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
                        png_buf.seek(0)
                        st.download_button(
                            "⬇️ Download SHAP force plot (PNG, 300 dpi)",
                            data=png_buf,
                            file_name="shap_force_plot.png",
                            mime="image/png",
                        )

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
                    st.error(f"Failed to generate SHAP plot: {e}")

# =========================
# Tab 2: Batch prediction
# =========================
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

        # Optional: user upload can update background too
        bg = df_in[FEATURES].dropna().head(max(bg_rows, 50)).copy()
        if len(bg) >= 20:
            st.session_state["shap_bg"] = bg
            st.success(f"SHAP background updated from uploaded CSV ({len(bg)} rows).")

        X_batch = df_in[FEATURES].copy()
        out = df_in.copy()
        out["pred_proba"] = model.predict_proba(X_batch)[:, 1]
        out["pred_label"] = model.predict(X_batch)

        st.write("Prediction preview:")
        st.dataframe(out.head(20), use_container_width=True)

        csv_bytes = out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="⬇️ Download prediction CSV",
            data=csv_bytes,
            file_name="svc_predictions.csv",
            mime="text/csv",
        )
