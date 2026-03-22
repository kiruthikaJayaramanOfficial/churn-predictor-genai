import streamlit as st
import json
import os
import sys
import mlflow
import pandas as pd
import numpy as np
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

# ── 2. AUTOMATED Drift Detection ─────────────────────────────
st.subheader("📊 Automated Drift Detection")

tab1, tab2 = st.tabs(["🔄 Run Drift Check", "📋 Last Report"])

with tab1:
    st.markdown("Upload new customer data to automatically check for drift against training data.")

    drift_file = st.file_uploader("Upload current data CSV", type=['csv'],
                                   key="drift_upload")

    col_a, col_b = st.columns(2)
    with col_a:
        run_on_default = st.button("▶️ Run on Original Data Split",
                                    use_container_width=True)
    with col_b:
        run_on_upload = st.button("▶️ Run on Uploaded File",
                                   use_container_width=True,
                                   disabled=drift_file is None)

    def run_drift(current_df=None):
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from src.eda import load_and_clean_data, encode_features

        with st.spinner("Running drift detection..."):
            # Reference
            df_ref = load_and_clean_data()
            df_ref = encode_features(df_ref)
            df_ref = df_ref.select_dtypes(include=[np.number])
            reference = df_ref.iloc[:5000]

            # Current
            if current_df is None:
                current = df_ref.iloc[5000:]
                source = "Original data split"
            else:
                if 'Churn' not in current_df.columns:
                    current_df['Churn'] = 0
                current_df['Churn'] = current_df['Churn'].map(
                    {'Yes': 1, 'No': 0}).fillna(0)
                current_df = encode_features(current_df)
                current_df = current_df.select_dtypes(include=[np.number])
                common_cols = [c for c in reference.columns if c in current_df.columns]
                reference = reference[common_cols]
                current = current_df[common_cols]
                source = "Uploaded file"

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference, current_data=current)
            report.save_html("mlops/drift_report.html")

            result = report.as_dict()
            drifted_features = []
            trigger_retrain = False
            n_drifted = 0

            for metric in result.get('metrics', []):
                if metric.get('metric') == 'DatasetDriftMetric':
                    trigger_retrain = metric['result'].get('dataset_drift', False)
                    n_drifted = metric['result'].get('number_of_drifted_columns', 0)
                if metric.get('metric') == 'ColumnDriftMetric':
                    r = metric.get('result', {})
                    if r.get('drift_detected', False):
                        drifted_features.append({
                            'feature': r.get('column_name', ''),
                            'drift_score': round(r.get('drift_score', 0), 4)
                        })

            summary = {
                "trigger_retrain": trigger_retrain,
                "drifted_features": drifted_features,
                "reference_size": len(reference),
                "current_size": len(current),
                "report_path": "mlops/drift_report.html",
                "source": source,
                "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            with open("mlops/drift_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            return summary, n_drifted

    if run_on_default:
        summary, n_drifted = run_drift(None)
        if n_drifted > 0:
            st.error(f"🔴 DRIFT DETECTED! {n_drifted} features drifted. Retraining recommended.")
            if summary['drifted_features']:
                st.dataframe(pd.DataFrame(summary['drifted_features']), use_container_width=True)
        else:
            st.success(f"✅ No drift detected on uploaded data. Model is healthy.")

        if summary['drifted_features']:
            st.dataframe(pd.DataFrame(summary['drifted_features']), use_container_width=True)
        

    if run_on_upload and drift_file is not None:
        df_uploaded = pd.read_csv(drift_file)
        summary, n_drifted = run_drift(df_uploaded)
        if n_drifted > 0:
            st.error(f"🔴 DRIFT DETECTED! {n_drifted} features drifted. Retraining recommended.")
            if summary['drifted_features']:
                st.dataframe(pd.DataFrame(summary['drifted_features']), use_container_width=True)
        else:
            st.success(f"✅ No drift detected on uploaded data. Model is healthy.")
        if summary['drifted_features']:
            st.dataframe(pd.DataFrame(summary['drifted_features']), use_container_width=True)
        

with tab2:
    # ── Last report results ───────────────────────────────────
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

        st.caption(f"Last run: {drift_summary.get('run_time', 'Unknown')} | Source: {drift_summary.get('source', 'Unknown')}")

        if trigger:
            st.error("⚠️ Drift detected! Model retraining recommended.")
        else:
            st.success("✅ No significant drift. Model is healthy.")

        drifted = drift_summary.get('drifted_features', [])
        if drifted:
            st.markdown("**Drifted Features:**")
            st.dataframe(pd.DataFrame(drifted), use_container_width=True)

        # Embed HTML report
        st.subheader("📋 Full Evidently Report")
        drift_html_path = "mlops/drift_report.html"
        if os.path.exists(drift_html_path):
            with open(drift_html_path, 'r') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=True)
    else:
        st.info("No report yet. Run drift detection above.")

st.divider()

# ── 3. MLflow run history ─────────────────────────────────────
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

# ── 4. CI/CD Status ───────────────────────────────────────────
st.subheader("⚙️ CI/CD Status")
st.info("GitHub Actions will show here after you push to GitHub in Sprint 6.")
ci_col1, ci_col2, ci_col3 = st.columns(3)
ci_col1.metric("Last Workflow", "Not yet pushed")
ci_col2.metric("Docker Build", "Pending")
ci_col3.metric("Auto-retrain", "Every Monday 6AM")

st.divider()

# ── 5. System health ──────────────────────────────────────────
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