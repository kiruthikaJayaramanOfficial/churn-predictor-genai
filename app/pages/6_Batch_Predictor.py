import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.explainability import load_champion, prepare_customer_input

st.set_page_config(page_title="Batch Predictor", page_icon="📦", layout="wide")
st.title("📦 Batch Predictor — Score all customers at once")

@st.cache_resource
def load_model():
    return load_champion()

model, feature_names = load_model()

st.markdown("Upload a CSV with Telco-format columns to score all customers at once.")

# ── Sample download ───────────────────────────────────────────
sample_data = pd.DataFrame([
    {'gender':'Male','SeniorCitizen':1,'Partner':'No','Dependents':'No',
     'tenure':2,'PhoneService':'Yes','MultipleLines':'No',
     'InternetService':'Fiber optic','OnlineSecurity':'No','OnlineBackup':'No',
     'DeviceProtection':'No','TechSupport':'No','StreamingTV':'No',
     'StreamingMovies':'No','Contract':'Month-to-month','PaperlessBilling':'Yes',
     'PaymentMethod':'Electronic check','MonthlyCharges':85.0,'TotalCharges':170.0},
    {'gender':'Female','SeniorCitizen':0,'Partner':'Yes','Dependents':'Yes',
     'tenure':36,'PhoneService':'Yes','MultipleLines':'Yes',
     'InternetService':'DSL','OnlineSecurity':'Yes','OnlineBackup':'Yes',
     'DeviceProtection':'Yes','TechSupport':'Yes','StreamingTV':'No',
     'StreamingMovies':'No','Contract':'Two year','PaperlessBilling':'No',
     'PaymentMethod':'Bank transfer (automatic)','MonthlyCharges':55.0,'TotalCharges':1980.0},
])
st.download_button("📥 Download Sample CSV", sample_data.to_csv(index=False),
                   "sample_customers.csv", "text/csv")

st.divider()

uploaded = st.file_uploader("📂 Upload Customer CSV", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"✅ Loaded {len(df):,} customers")

    with st.spinner("Scoring all customers..."):
        probs = []
        for _, row in df.iterrows():
            try:
                input_df = prepare_customer_input(row.to_dict(), feature_names)
                prob = model.predict_proba(input_df)[0][1]
                probs.append(round(prob, 4))
            except:
                probs.append(None)

        df['churn_probability'] = probs
        df['risk_tier'] = df['churn_probability'].apply(
            lambda p: 'High' if p > 0.7 else ('Medium' if p > 0.4 else 'Low')
            if p is not None else 'Unknown'
        )

    # ── Metrics ───────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", len(df))
    c2.metric("High Risk", len(df[df['risk_tier']=='High']))
    c3.metric("Medium Risk", len(df[df['risk_tier']=='Medium']))
    c4.metric("Low Risk", len(df[df['risk_tier']=='Low']))

    # ── Pie chart ─────────────────────────────────────────────
    risk_counts = df['risk_tier'].value_counts().reset_index()
    risk_counts.columns = ['Risk Tier', 'Count']
    fig = px.pie(risk_counts, values='Count', names='Risk Tier',
                 color='Risk Tier',
                 color_discrete_map={'High':'#ef4444','Medium':'#f97316','Low':'#22c55e'},
                 title="Customer Risk Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Top 10 highest risk ───────────────────────────────────
    st.subheader("🔴 Top 10 Highest Risk Customers")
    top10 = df.nlargest(10, 'churn_probability')[
        ['churn_probability', 'risk_tier', 'tenure',
         'MonthlyCharges', 'Contract'] +
        [c for c in ['gender', 'InternetService'] if c in df.columns]
    ]
    st.dataframe(top10, use_container_width=True)

    st.divider()

    # ── Download results ──────────────────────────────────────
    st.download_button(
        "📥 Download Scored CSV",
        df.to_csv(index=False),
        "scored_customers.csv",
        "text/csv",
        use_container_width=True
    )
else:
    st.info("👆 Upload a CSV to get started. Use the sample CSV above as a template.")