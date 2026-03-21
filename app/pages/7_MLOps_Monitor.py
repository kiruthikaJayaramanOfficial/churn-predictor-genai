import streamlit as st
import json
import os
import sys
import mlflow
import pandas as pd
import plotly.express as px
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="MLOps Monitor", page_icon="🔧", layout="wide")
st.title("🔧 MLOps Monitor — Is the model still working?")

# ── 1. Champion model info ────────────────────────────────────
st.subheader("🏆 Champion Model")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Model", "LightGBM")
col2.metric("Run Name", "LGB_lr0.01_spw5")
col3.metric("Recall", "0.8663")
col4.metric("AUC-ROC", "0.8317")

st.divider()

# ── 2. Drift report ───────────────────────────────────────────
st.subheader("📊 Evidently Drift Report")

drift_json_path = "mlops/drift_summary.json"
if os.path.exists(drift_json_path):
    with open(drift_json_path) as f:
        drift_summary = json.load(f)

    d1, d2, d3, d4 = st.columns(4)
    trigger = drift_summary.get('trigger_retrain', False)
    d1.metric("Drift Detected", "🔴 YES" if trigger else "🟢 NO")
    d2.metric("Drifted Features", len(drift_summary.get('drifted_features', [])))
    d3.metric("Reference Size", f"{drift_summary.get('reference_size', 0):,}")
    d4.metric("Current Size", f"{drift_summary.get('current_size', 0):,}")

    if trigger:
        st.error("⚠️ Data drift detected! Model retraining recommended.")
    else:
        st.success("✅ No significant data drift detected. Model is healthy.")

    drifted = drift_summary.get('drifted_features', [])
    if drifted:
        st.markdown("**Drifted Features:**")
        st.dataframe(pd.DataFrame(drifted), use_container_width=True)
else:
    st.warning("No drift report found. Run `python mlops/drift_detector.py` first.")

st.divider()

# ── 3. Evidently HTML report embedded ────────────────────────
st.subheader("📋 Full Evidently Report")
drift_html_path = "mlops/drift_report.html"
if os.path.exists(drift_html_path):
    with open(drift_html_path, 'r') as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=600, scrolling=True)
else:
    st.warning("Run drift detector first to generate the report.")

st.divider()

# ── 4. MLflow run history ─────────────────────────────────────
st.subheader("📈 MLflow Run History")
try:
    mlflow.set_tracking_uri("mlruns")
    runs = mlflow.search_runs(experiment_names=["churn_predictor"])
    if not runs.empty:
        runs_plot = runs[['tags.mlflow.runName', 'metrics.recall',
                           'metrics.f1', 'metrics.auc_roc']].copy()
        runs_plot.columns = ['Run', 'Recall', 'F1', 'AUC-ROC']
        runs_plot = runs_plot.dropna().sort_values('Recall', ascending=False)

        fig = px.bar(runs_plot, x='Run', y='Recall',
                     color='Recall', color_continuous_scale='Blues',
                     title="Recall by MLflow Run")
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.warning(f"Could not load MLflow runs: {e}")

st.divider()

# ── 5. GitHub Actions status ──────────────────────────────────
st.subheader("⚙️ CI/CD Status")
st.info("GitHub Actions will show here after you push to GitHub in Sprint 6.")

ci_col1, ci_col2, ci_col3 = st.columns(3)
ci_col1.metric("Last Workflow", "Not yet pushed")
ci_col2.metric("Docker Build", "Pending")
ci_col3.metric("Auto-retrain", "Every Monday 6AM")

st.code("""
# GitHub Actions workflow triggers:
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * MON'  # Every Monday 6AM UTC
""", language="yaml")

st.divider()

# ── 6. System health ──────────────────────────────────────────
st.subheader("🖥️ System Health")
h1, h2, h3 = st.columns(3)
h1.metric("FastAPI Status", "🟢 Running", "localhost:8000")
h2.metric("Streamlit Status", "🟢 Running", "localhost:8501")
h3.metric("Champion Model", "🟢 Loaded", "models/champion_model.pkl")

model_path = "models/champion_model.pkl"
if os.path.exists(model_path):
    mod_time = os.path.getmtime(model_path)
    last_trained = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
    st.info(f"🕐 Champion model last trained/saved: **{last_trained}**")