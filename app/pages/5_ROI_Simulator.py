import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.explainability import load_champion, prepare_customer_input

st.set_page_config(page_title="ROI Simulator", page_icon="💰", layout="wide")
st.title("💰 ROI Simulator — What if we change their plan?")
st.markdown("Adjust the sliders to simulate retention interventions and see how churn probability changes.")

@st.cache_resource
def load_model():
    return load_champion()

model, feature_names = load_model()

# ── Base customer (from session or defaults) ──────────────────
if 'last_prediction' in st.session_state:
    base_customer = st.session_state['last_prediction']['customer_info']
    base_prob = st.session_state['last_prediction']['prob']
    st.success("✅ Using customer from Live Predictor")
else:
    base_customer = {
        'gender': 'Male', 'SeniorCitizen': 1, 'Partner': 'No',
        'Dependents': 'No', 'tenure': 2, 'PhoneService': 'Yes',
        'MultipleLines': 'No', 'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No', 'OnlineBackup': 'No',
        'DeviceProtection': 'No', 'TechSupport': 'No',
        'StreamingTV': 'No', 'StreamingMovies': 'No',
        'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.0, 'TotalCharges': 170.0
    }
    base_df = prepare_customer_input(base_customer, feature_names)
    base_prob = model.predict_proba(base_df)[0][1]
    st.info("ℹ️ Using default high-risk customer. Go to Live Predictor to use your own.")

st.divider()

# ── Intervention sliders ──────────────────────────────────────
st.subheader("🎛️ Retention Interventions")

col1, col2 = st.columns(2)

with col1:
    tenure_boost = st.slider("📅 Tenure boost (months)", 0, 24, 0, 1,
                              help="Simulate customer staying longer")
    charge_reduction = st.slider("💵 Monthly charge reduction (%)", 0, 50, 0, 5,
                                  help="Discount on monthly charges")

with col2:
    contract_upgrade = st.checkbox("📋 Upgrade to One-year contract",
                                    help="Switch from month-to-month to 1-year")
    add_tech_support = st.checkbox("🔧 Add Tech Support",
                                    help="Bundle tech support service")
    add_security = st.checkbox("🔒 Add Online Security",
                                help="Bundle online security service")

# ── Simulate modified customer ────────────────────────────────
modified = base_customer.copy()
modified['tenure'] = base_customer['tenure'] + tenure_boost
modified['MonthlyCharges'] = base_customer['MonthlyCharges'] * (1 - charge_reduction / 100)
modified['TotalCharges'] = modified['MonthlyCharges'] * modified['tenure']

if contract_upgrade:
    modified['Contract'] = 'One year'
if add_tech_support:
    modified['TechSupport'] = 'Yes'
if add_security:
    modified['OnlineSecurity'] = 'Yes'

modified_df = prepare_customer_input(modified, feature_names)
new_prob = model.predict_proba(modified_df)[0][1]
prob_delta = base_prob - new_prob
avg_customer_value = modified['MonthlyCharges'] * 12
revenue_saved = prob_delta * avg_customer_value

st.divider()

# ── Results ───────────────────────────────────────────────────
st.subheader("📊 Impact of Interventions")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Original Churn Prob", f"{base_prob*100:.1f}%")
c2.metric("New Churn Prob", f"{new_prob*100:.1f}%",
          delta=f"{-prob_delta*100:.1f}%",
          delta_color="inverse")
c3.metric("Risk Reduction", f"{prob_delta*100:.1f}pp")
c4.metric("Est. Revenue Saved", f"₹{revenue_saved:,.0f}/yr")

# ── Gauge comparison ──────────────────────────────────────────
g1, g2 = st.columns(2)

def make_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40],  'color': '#dcfce7'},
                {'range': [40, 70], 'color': '#fef9c3'},
                {'range': [70, 100],'color': '#fee2e2'},
            ]
        }
    ))
    fig.update_layout(height=250)
    return fig

with g1:
    st.plotly_chart(make_gauge(base_prob, "Before Intervention", "#ef4444"),
                    use_container_width=True)
with g2:
    new_color = "#ef4444" if new_prob > 0.7 else "#f97316" if new_prob > 0.4 else "#22c55e"
    st.plotly_chart(make_gauge(new_prob, "After Intervention", new_color),
                    use_container_width=True)

st.divider()

# ── Business case bar chart ───────────────────────────────────
st.subheader("💼 Business Case")
scenarios = {
    'No Action': base_prob * base_customer['MonthlyCharges'] * 12,
    'Charge -10%': None,
    'Charge -20%': None,
    'Contract Upgrade': None,
    'All Interventions': None
}

def sim_prob(tenure_add=0, charge_red=0, contract='Month-to-month',
             tech='No', security='No'):
    s = base_customer.copy()
    s['tenure'] = base_customer['tenure'] + tenure_add
    s['MonthlyCharges'] = base_customer['MonthlyCharges'] * (1 - charge_red)
    s['TotalCharges'] = s['MonthlyCharges'] * s['tenure']
    s['Contract'] = contract
    s['TechSupport'] = tech
    s['OnlineSecurity'] = security
    df = prepare_customer_input(s, feature_names)
    return model.predict_proba(df)[0][1]

revenue_scenarios = {
    'No Action':         sim_prob() * base_customer['MonthlyCharges'] * 12,
    'Charge -10%':       sim_prob(charge_red=0.10) * base_customer['MonthlyCharges'] * 0.90 * 12,
    'Charge -20%':       sim_prob(charge_red=0.20) * base_customer['MonthlyCharges'] * 0.80 * 12,
    'Contract Upgrade':  sim_prob(contract='One year') * base_customer['MonthlyCharges'] * 12,
    'All Interventions': sim_prob(tenure_add=6, charge_red=0.15, contract='One year',
                                   tech='Yes', security='Yes') * base_customer['MonthlyCharges'] * 0.85 * 12
}

fig_bar = go.Figure(go.Bar(
    x=list(revenue_scenarios.keys()),
    y=list(revenue_scenarios.values()),
    marker_color=['#ef4444','#f97316','#f59e0b','#22c55e','#3b82f6'],
    text=[f"₹{v:,.0f}" for v in revenue_scenarios.values()],
    textposition='outside'
))
fig_bar.update_layout(
    title="Expected Revenue at Risk by Scenario (lower = better retention)",
    yaxis_title="Expected Revenue at Risk (₹/yr)",
    height=400
)
st.plotly_chart(fig_bar, use_container_width=True)