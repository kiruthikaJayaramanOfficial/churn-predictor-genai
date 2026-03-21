import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.langchain_agent import get_churn_playbook

st.set_page_config(page_title="Churn Playbook Agent", page_icon="🤖", layout="wide")
st.title("🤖 Churn Playbook Agent — WHY + what to do")

if 'last_prediction' not in st.session_state:
    st.warning("⚠️ No prediction found. Go to **Live Predictor** first, enter a customer and click Predict.")
    st.stop()

pred = st.session_state['last_prediction']
prob = pred['prob']
shap_result = pred['shap_result']
customer_info = pred['customer_info']
risk = pred['risk']

# ── Summary ───────────────────────────────────────────────────
col1, col2 = st.columns(2)
col1.metric("Churn Probability", f"{prob*100:.1f}%")
col2.metric("Risk Level", risk)

st.divider()

# ── SHAP summary ──────────────────────────────────────────────
st.subheader("📊 Top Churn Drivers (from SHAP)")
top_pos = shap_result['top_positive']
top_neg = shap_result['top_negative']

c1, c2 = st.columns(2)
with c1:
    st.markdown("**🔴 Factors INCREASING churn risk:**")
    for item in top_pos:
        st.markdown(f"- `{item['feature']}` = {item['feature_value']:.2f} → impact: **+{item['shap_value']:.3f}**")

with c2:
    st.markdown("**🟢 Factors REDUCING churn risk:**")
    for item in top_neg:
        st.markdown(f"- `{item['feature']}` = {item['feature_value']:.2f} → impact: **{item['shap_value']:.3f}**")

st.divider()

# ── Generate playbook ─────────────────────────────────────────
st.subheader("📋 AI-Generated Churn Playbook")

if st.button("🚀 Generate Churn Playbook", type="primary", use_container_width=True):
    st.markdown("---")
    placeholder = st.empty()
    full_text = ""

    def stream_to_ui(token):
        global full_text
        full_text += token
        placeholder.markdown(full_text + "▌")

    with st.spinner("Generating playbook..."):
        result = get_churn_playbook(
            churn_probability=prob,
            shap_result=shap_result,
            customer_info=customer_info,
            stream_callback=stream_to_ui
        )

    placeholder.markdown(result)
    st.success("✅ Playbook generated!")
    st.balloons()