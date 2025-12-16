
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="SVC Risk Predictor", layout="wide")

FEATURES = [
    'Visual impairment',
    'Clival invasion',
    'Hardy D-E',
    'p53 positivity',
    'Ki-67≥3%',
    'High-risk subtype',
    'Residual tumor'
]
TARGET = "event"

DEFAULT_MODEL_PATH = r"C:\\Users\\86155\\Desktop\\best_svc.pkl"

st.title("🧠 SVC 临床风险预测交互网页")
st.caption("输入病人特征或上传 CSV，输出 SVC 预测概率与类别。")

def ensure_features(df: pd.DataFrame, features: list[str]):
    missing = [c for c in features if c not in df.columns]
    if missing:
        return False, f"缺少特征列：{missing}"
    return True, ""

def safe_auc(y_true, proba):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, proba)

st.sidebar.header("⚙️ 模型加载")
model_path = st.sidebar.text_input("模型路径 (best_svc.pkl)", value=DEFAULT_MODEL_PATH)

try:
    model = joblib.load(model_path)
    st.sidebar.success("模型加载成功 ✅")
except Exception as e:
    st.sidebar.error(f"模型加载失败：{e}")
    st.stop()

tab1, tab2 = st.tabs(["🧍 单病例预测", "📄 批量预测 (CSV)"])

with tab1:
    st.subheader("单病例输入 → 风险预测")

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
        'Visual impairment': vi,
        'Clival invasion': ci,
        'Hardy D-E': hardy,
        'p53 positivity': p53,
        'Ki-67≥3%': ki67,
        'High-risk subtype': subtype,
        'Residual tumor': residual
    }])

    st.write("输入特征：")
    st.dataframe(X_one, use_container_width=True)

    thresh = st.slider("阈值：pred_proba ≥ 阈值 判为高风险(1)", 0.0, 1.0, 0.5, 0.01)

    if st.button("🔮 预测"):
        proba = float(model.predict_proba(X_one)[:, 1][0])
        pred_by_thresh = int(proba >= thresh)

        m1, m2 = st.columns(2)
        m1.metric("pred_proba (正类概率)", f"{proba:.4f}")
        m2.metric(f"pred_label (按阈值 {thresh:.2f})", f"{pred_by_thresh}")

        if pred_by_thresh == 1:
            st.error("结果：高风险 (1)")
        else:
            st.success("结果：低风险 (0)")

with tab2:
    st.subheader("上传 CSV → 批量预测 → 下载结果")
    st.info(f"CSV 必须包含 7 个特征列：{FEATURES}")

    uploaded = st.file_uploader("上传 CSV 文件", type=["csv"])
    if uploaded is not None:
        df_in = pd.read_csv(uploaded)

        ok, msg = ensure_features(df_in, FEATURES)
        if not ok:
            st.error(msg)
            st.stop()

        X_batch = df_in[FEATURES].copy()
        proba = model.predict_proba(X_batch)[:, 1]
        pred = model.predict(X_batch)

        out = df_in.copy()
        out["pred_label"] = pred
        out["pred_proba"] = proba

        st.write("预测结果预览：")
        st.dataframe(out.head(20), use_container_width=True)

        csv_bytes = out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="⬇️ 下载预测结果 CSV",
            data=csv_bytes,
            file_name="svc_predictions.csv",
            mime="text/csv"
        )
