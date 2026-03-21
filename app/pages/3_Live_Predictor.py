import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.explainability import load_champion, get_shap_for_customer, prepare_customer_input

st.set_page_config(page_title="Live Predictor", page_icon="🎯", layout="wide")
st.title("🎯 Live Predictor — Will this customer churn?")

@st.cache_resource
def load_model():
    return load_champion()

model, feature_names = load_model()

# ── Input form ────────────────────────────────────────────────
st.subheader("📋 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])

with col2:
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

with col3:
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
    total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                     float(monthly_charges * tenure))

st.divider()

if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    input_dict = {
        'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract,
        'PaperlessBilling': paperless_billing, 'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }

    customer_df = prepare_customer_input(input_dict, feature_names)
    prob = model.predict_proba(customer_df)[0][1]
    risk = "🔴 HIGH" if prob > 0.7 else "🟡 MEDIUM" if prob > 0.4 else "🟢 LOW"
    risk_color = "#ef4444" if prob > 0.7 else "#f97316" if prob > 0.4 else "#22c55e"

    # ── Results row ───────────────────────────────────────────
    r1, r2, r3 = st.columns(3)
    r1.metric("Churn Probability", f"{prob*100:.1f}%")
    r2.metric("Risk Tier", risk)
    r3.metric("Tenure", f"{tenure} months")

    # ── Gauge chart ───────────────────────────────────────────
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Churn Probability %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': risk_color},
            'steps': [
                {'range': [0, 40],  'color': '#dcfce7'},
                {'range': [40, 70], 'color': '#fef9c3'},
                {'range': [70, 100],'color': '#fee2e2'},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prob * 100
            }
        }
    ))
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ── SHAP waterfall ────────────────────────────────────────
    st.subheader("🔍 Why this prediction? (SHAP Feature Impact)")
    shap_result = get_shap_for_customer(customer_df, model, feature_names)
    all_impacts = shap_result['all_impacts'].head(10)

    colors = ['#ef4444' if v > 0 else '#22c55e' for v in all_impacts['shap_value']]
    fig_shap = go.Figure(go.Bar(
        x=all_impacts['shap_value'],
        y=all_impacts['feature'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:.3f}" for v in all_impacts['shap_value']],
        textposition='outside'
    ))
    fig_shap.update_layout(
        title="Top 10 Feature Impacts (Red = increases churn risk, Green = reduces)",
        xaxis_title="SHAP Value",
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # Store in session for Page 4
    st.session_state['last_prediction'] = {
        'prob': prob, 'shap_result': shap_result,
        'customer_info': input_dict, 'risk': risk
    }
    st.info("💡 Go to **Churn Playbook Agent** in the sidebar to get AI-powered retention actions!")