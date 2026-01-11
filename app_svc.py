import io
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

# =========================
# Config (match notebook)
# =========================
st.set_page_config(page_title="SVC Risk Predictor (Notebook-matched)", layout="wide")

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
BG_RANDOM_STATE = 0  # notebook uses random_state=0
DEFAULT_BG_ROWS = 50
DEFAULT_NSAMPLES = 200

# Feature label mapping (same as your paper-style)
FEATURE_LABEL_MAP = {
    "Visual impairment": "Visual impairment",
    "Clival invasion": "Clival invasion",
    "Hardy D-E": "Hardy grade D–E",
    "p53 positivity": "p53 positivity",
    "Ki-67≥3%": "Ki-67 ≥ 3%",
    "High-risk subtype": "High-risk subtype",
    "Residual tumor": "Residual tumor",
}

# =========================
# Typography (Times New Roman)
# =========================
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["font.size"] = 9

st.title("🧠 SVC Clinical Risk Prediction Dashboard (Notebook-Matched)")
st.caption("This app reproduces your notebook logic: X_test → quantile index → KernelExplainer(background sample) → force plot.")

# =========================
# Helpers
# =========================
def ensure_features(df: pd.DataFrame, features: list[str]):
    missing = [c for c in features if c not in df.columns]
    if missing:
        return False, f"Missing feature columns: {missing}"
    return True, ""

def scalar_expected_value(ev) -> float:
    return float(np.array(ev).ravel()[0])

def format_feature_label(name: str) -> str:
    return FEATURE_LABEL_MAP.get(name, name)

def clean_and_style_forceplot_texts(
    ax: plt.Axes,
    fx: float,
    baseline: float,
    *,
    label_fontsize: int = 8,
    title_fontsize: int = 11,
):
    # remove internal f(x) / base value texts
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

    # replace biggest numeric with true fx (your notebook behavior)
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
            mapped = format_feature_label(left_clean)
            right = re.sub(r"(-?\d+)\.0\b", r"\1", right)
            txt.set_text(f"{mapped} = {right}")

    # title
    ax.set_title(f"f(x) = {fx:.2f}, baseline = {baseline:.2f}", fontsize=title_fontsize, pad=8)

    # uniform text font size
    for txt in ax.texts:
        txt.set_fontsize(label_fontsize)

def make_force_plot_figure(
    *,
    base_value: float,
    shap_values_1d: np.ndarray,
    x_row: pd.Series,
    fx: float,
    baseline: float,
    display: bool,
) -> plt.Figure:
    shap.force_plot(
        base_value=base_value,
        shap_values=shap_values_1d,
        features=x_row,
        feature_names=x_row.index.tolist(),
        matplotlib=True,
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(12, 2.5)

    # IMPORTANT: Streamlit display should not be huge dpi; exports will set dpi explicitly
    fig.set_dpi(150 if display else 600)

    ax = plt.gca()
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("Predicted probability", fontsize=9)

    clean_and_style_forceplot_texts(ax=ax, fx=fx, baseline=baseline, label_fontsize=8, title_fontsize=11)
    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout(pad=1.5)
    return fig

# =========================
# Sidebar inputs (match notebook)
# =========================
st.sidebar.header("⚙️ Notebook-matched settings")

model_path = st.sidebar.text_input("Model path", value=DEFAULT_MODEL_PATH)
try:
    model = joblib.load(model_path)
    st.sidebar.success("Model loaded ✅")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

st.sidebar.markdown("---")
x_test_file = st.sidebar.file_uploader("Upload X_test.csv (REQUIRED to match notebook)", type=["csv"])

bg_rows = st.sidebar.number_input("background size (bg_rows)", min_value=10, max_value=500, value=DEFAULT_BG_ROWS, step=10)
nsamples = st.sidebar.number_input("nsamples", min_value=50, max_value=2000, value=DEFAULT_NSAMPLES, step=50)

q_high = st.sidebar.slider("High quantile (notebook=0.9)", 0.5, 0.99, 0.90, 0.01)
q_low = st.sidebar.slider("Low quantile (notebook=0.1)", 0.01, 0.5, 0.10, 0.01)

show_high = st.sidebar.checkbox("Show high-quantile patient force plot", value=True)
show_low = st.sidebar.checkbox("Show low-quantile patient force plot", value=True)

show_manual = st.sidebar.checkbox("Also explain a manual row index", value=False)
manual_idx = st.sidebar.number_input("manual row index", min_value=0, value=0, step=1)

# =========================
# Load X_test
# =========================
if x_test_file is None:
    st.warning("请在左侧上传 X_test.csv（必须与 notebook 中 X_test 完全相同，才能做到一致）。")
    st.stop()

try:
    X_test = pd.read_csv(x_test_file)
except Exception as e:
    st.error(f"Cannot read X_test.csv: {e}")
    st.stop()

ok, msg = ensure_features(X_test, FEATURES)
if not ok:
    st.error(msg)
    st.stop()

# Ensure same column order as notebook intended
X_test = X_test[FEATURES].copy()

st.subheader("Dataset check")
st.write(f"X_test shape: {X_test.shape}")
st.dataframe(X_test.head(5), use_container_width=True)

# =========================
# Build background + explainer (match notebook)
# =========================
# Notebook: background = shap.sample(X_test, 50, random_state=0)
background = shap.sample(X_test, int(bg_rows), random_state=BG_RANDOM_STATE)

# Notebook-style f_prob (IMPORTANT: wrap numpy into DataFrame with correct columns)
def f_prob(X):
    X_df = pd.DataFrame(X, columns=FEATURES)
    return model.predict_proba(X_df)[:, 1]

@st.cache_resource
def build_explainer(bg_np: np.ndarray, seed: int):
    _ = seed
    return shap.KernelExplainer(f_prob, bg_np)

bg_np = background.to_numpy(dtype=float)
explainer = build_explainer(bg_np, BG_RANDOM_STATE)

base_value = scalar_expected_value(explainer.expected_value)
baseline = base_value  # notebook baseline semantics

# =========================
# Compute probs & select indices (match notebook)
# =========================
probs = model.predict_proba(X_test)[:, 1]
sorted_idx = np.argsort(probs)

idx_high = int(sorted_idx[int(q_high * len(sorted_idx))])
idx_low = int(sorted_idx[int(q_low * len(sorted_idx))])

st.subheader("Notebook-matched selection")
c1, c2, c3 = st.columns(3)
c1.metric("baseline (expected_value)", f"{baseline:.4f}")
c2.metric(f"idx_high (q={q_high:.2f})", f"{idx_high} (p={probs[idx_high]:.4f})")
c3.metric(f"idx_low (q={q_low:.2f})", f"{idx_low} (p={probs[idx_low]:.4f})")

# =========================
# Compute SHAP values for selected rows (match notebook)
# =========================
def compute_shap_for_rows(rows: pd.DataFrame) -> np.ndarray:
    # make deterministic-ish
    np.random.seed(BG_RANDOM_STATE)
    sv = explainer.shap_values(rows.to_numpy(dtype=float), nsamples=int(nsamples))
    if isinstance(sv, list):
        sv_arr = np.asarray(sv[0])
    else:
        sv_arr = np.asarray(sv)
    return sv_arr  # shape: (n_rows, n_features)

# Compute shap for required rows (only once)
rows_to_explain = []
row_names = []
if show_high:
    rows_to_explain.append(X_test.iloc[idx_high])
    row_names.append(f"High quantile (idx={idx_high}, p={probs[idx_high]:.4f})")
if show_low:
    rows_to_explain.append(X_test.iloc[idx_low])
    row_names.append(f"Low quantile (idx={idx_low}, p={probs[idx_low]:.4f})")
if show_manual:
    if manual_idx < 0 or manual_idx >= len(X_test):
        st.error("manual index out of range.")
        st.stop()
    rows_to_explain.append(X_test.iloc[int(manual_idx)])
    row_names.append(f"Manual (idx={int(manual_idx)}, p={probs[int(manual_idx)]:.4f})")

if len(rows_to_explain) == 0:
    st.info("No plots selected.")
    st.stop()

rows_df = pd.DataFrame(rows_to_explain, columns=FEATURES)
sv_arr = compute_shap_for_rows(rows_df)

# =========================
# Render plots + downloads
# =========================
st.subheader("Model explanation (SHAP force plots)")

for i, title in enumerate(row_names):
    x_row = rows_df.iloc[i]
    fx = float(model.predict_proba(pd.DataFrame([x_row], columns=FEATURES))[:, 1][0])
    sv_1d = sv_arr[i]

    st.markdown(f"### {title}")
    st.write("Input row:")
    st.dataframe(pd.DataFrame([x_row]), use_container_width=True)

    fig_disp = make_force_plot_figure(
        base_value=base_value,
        shap_values_1d=sv_1d,
        x_row=x_row,
        fx=fx,
        baseline=baseline,
        display=True,
    )
    st.pyplot(fig_disp, use_container_width=True, clear_figure=False)

    # Export high-dpi images (match your notebook saving)
    # TIFF 600 dpi
    tiff_buf = io.BytesIO()
    fig_disp.savefig(
        tiff_buf,
        format="tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.02,
    )
    tiff_buf.seek(0)
    st.download_button(
        f"⬇️ Download TIFF (600 dpi) - {title}",
        data=tiff_buf,
        file_name=f"force_{i+1}.tiff",
        mime="image/tiff",
    )

    # PNG 300 dpi
    png_buf = io.BytesIO()
    fig_disp.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
    png_buf.seek(0)
    st.download_button(
        f"⬇️ Download PNG (300 dpi) - {title}",
        data=png_buf,
        file_name=f"force_{i+1}.png",
        mime="image/png",
    )

    # PDF
    pdf_buf = io.BytesIO()
    fig_disp.savefig(pdf_buf, format="pdf", bbox_inches="tight")
    pdf_buf.seek(0)
    st.download_button(
        f"⬇️ Download PDF - {title}",
        data=pdf_buf,
        file_name=f"force_{i+1}.pdf",
        mime="application/pdf",
    )

    plt.close(fig_disp)
