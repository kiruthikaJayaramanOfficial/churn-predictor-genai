import streamlit as st
import mlflow

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Model Comparison", page_icon="🏆", layout="wide")
st.title("🏆 Model Comparison — Which model wins?")

# ── Load all MLflow runs ──────────────────────────────────────
@st.cache_data
def load_runs():
    # Read directly from local mlruns folder — no HTTP needed
    mlflow.set_tracking_uri("mlruns")
    runs = mlflow.search_runs(experiment_names=["churn_predictor"])
    runs = runs[['tags.mlflow.runName',
                 'metrics.recall', 'metrics.precision',
                 'metrics.f1', 'metrics.auc_roc',
                 'params.model']].copy()
    runs.columns = ['Run Name', 'Recall', 'Precision', 'F1', 'AUC-ROC', 'Model Type']
    runs = runs.dropna(subset=['Recall'])
    runs = runs.sort_values('Recall', ascending=False).reset_index(drop=True)
    runs['Recall']    = runs['Recall'].round(4)
    runs['Precision'] = runs['Precision'].round(4)
    runs['F1']        = runs['F1'].round(4)
    runs['AUC-ROC']   = runs['AUC-ROC'].round(4)
    return runs
runs = load_runs()

# ── Champion banner ───────────────────────────────────────────
champion = runs.iloc[0]
st.success(f"🥇 **Champion Model: {champion['Run Name']}** — Recall: {champion['Recall']} | F1: {champion['F1']} | AUC-ROC: {champion['AUC-ROC']}")

st.divider()

# ── Runs table ────────────────────────────────────────────────
st.subheader("📋 All 12 MLflow Runs — Sorted by Recall")

def highlight_champion(row):
    if row.name == 0:
        return ['background-color: #fef9c3'] * len(row)
    return [''] * len(row)

styled = runs.style.apply(highlight_champion, axis=1).format({
    'Recall': '{:.4f}', 'Precision': '{:.4f}',
    'F1': '{:.4f}', 'AUC-ROC': '{:.4f}'
})
st.dataframe(styled, use_container_width=True, height=420)

st.divider()

# ── Bar chart: Recall by model ────────────────────────────────
st.subheader("📊 Recall by Run")
colors = ['#f59e0b' if i == 0 else '#3b82f6' for i in range(len(runs))]
fig1 = go.Figure(go.Bar(
    x=runs['Run Name'],
    y=runs['Recall'],
    marker_color=colors,
    text=runs['Recall'],
    texttemplate='%{text:.3f}',
    textposition='outside'
))
fig1.update_layout(
    xaxis_tickangle=-45,
    yaxis_title="Recall",
    yaxis_range=[0, 1.05],
    showlegend=False,
    height=450
)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# ── Multi-metric comparison ───────────────────────────────────
st.subheader("📈 Multi-Metric Comparison")
metrics = ['Recall', 'Precision', 'F1', 'AUC-ROC']
fig2 = go.Figure()
colors_m = ['#ef4444', '#3b82f6', '#22c55e', '#f97316']
for metric, color in zip(metrics, colors_m):
    fig2.add_trace(go.Bar(
        name=metric,
        x=runs['Run Name'],
        y=runs[metric],
        marker_color=color
    ))
fig2.update_layout(
    barmode='group',
    xaxis_tickangle=-45,
    yaxis_title="Score",
    yaxis_range=[0, 1.05],
    height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Model type summary ────────────────────────────────────────
st.subheader("🔍 Best Recall by Model Type")
if 'Model Type' in runs.columns:
    best_by_type = runs.groupby('Model Type')['Recall'].max().reset_index()
    best_by_type = best_by_type.sort_values('Recall', ascending=False)
    fig3 = px.bar(best_by_type, x='Model Type', y='Recall',
                  color='Recall', color_continuous_scale='Blues',
                  text='Recall')
    fig3.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig3.update_layout(yaxis_range=[0, 1.05], coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)