import streamlit as st

st.set_page_config(page_title="Churn Predictor", page_icon="🔮", layout="wide")

st.title("🔮 Churn Predictor + GenAI Agent")
st.markdown("""
### P3 — XGBoost + LangChain + MLflow + SHAP

Navigate using the sidebar to explore:

| Page | Description |
|------|-------------|
| 📊 EDA Dashboard | Understand churn patterns |
| 🏆 Model Comparison | 12-run MLflow results |
| 🎯 Live Predictor | Real-time churn prediction |
| 🤖 Churn Playbook Agent | AI-generated retention plan |
| 💰 ROI Simulator | Business impact calculator |
| 📦 Batch Predictor | Score all customers at once |
| 🔧 MLOps Monitor | Drift + CI/CD health |
""")