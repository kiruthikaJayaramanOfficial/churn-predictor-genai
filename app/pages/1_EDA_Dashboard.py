import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.eda import load_and_clean_data

st.set_page_config(page_title="EDA Dashboard", page_icon="📊", layout="wide")
st.title("📊 EDA Dashboard — What does churn look like?")

df = load_and_clean_data()

# ── Top metrics row ──────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(df):,}")
col2.metric("Churned Customers", f"{df['Churn'].sum():,}")
col3.metric("Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
col4.metric("Avg Monthly Charges", f"₹{df['MonthlyCharges'].mean():.2f}")

st.divider()

# ── Chart 1: Churn rate by contract type ─────────────────────
st.subheader("1️⃣ Churn Rate by Contract Type")
contract_churn = df.groupby('Contract')['Churn'].mean().reset_index()
contract_churn['Churn %'] = (contract_churn['Churn'] * 100).round(1)
fig1 = px.bar(contract_churn, x='Contract', y='Churn %',
              color='Contract',
              color_discrete_sequence=['#ef4444','#f97316','#22c55e'],
              text='Churn %')
fig1.update_traces(texttemplate='%{text}%', textposition='outside')
fig1.update_layout(showlegend=False, yaxis_title="Churn Rate (%)")
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# ── Chart 2: Tenure histogram coloured by churn ──────────────
st.subheader("2️⃣ Tenure Distribution by Churn Status")
df['Churn Label'] = df['Churn'].map({1: 'Churned', 0: 'Stayed'})
fig2 = px.histogram(df, x='tenure', color='Churn Label',
                    nbins=50, barmode='overlay',
                    color_discrete_map={'Churned': '#ef4444', 'Stayed': '#22c55e'},
                    labels={'tenure': 'Tenure (months)'})
fig2.update_layout(yaxis_title="Customer Count")
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Chart 3: Monthly charges violin plot ─────────────────────
st.subheader("3️⃣ Monthly Charges Distribution by Churn")
fig3 = px.violin(df, y='MonthlyCharges', x='Churn Label',
                 color='Churn Label',
                 color_discrete_map={'Churned': '#ef4444', 'Stayed': '#22c55e'},
                 box=True, points='outliers')
fig3.update_layout(xaxis_title="Churn Status", yaxis_title="Monthly Charges")
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Chart 4: Correlation heatmap ─────────────────────────────
st.subheader("4️⃣ Correlation Heatmap")
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn']
corr = df[numeric_cols].corr().round(2)
fig4 = px.imshow(corr, text_auto=True, aspect="auto",
                 color_continuous_scale='RdBu_r',
                 zmin=-1, zmax=1)
st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── Chart 5: Top 5 churn drivers ─────────────────────────────
st.subheader("5️⃣ Top 5 Churn Drivers")
drivers = {
    'Month-to-month contract': 42.7,
    'No online security': 41.8,
    'Fiber optic internet': 41.9,
    'Tenure < 12 months': 47.4,
    'No tech support': 41.6
}
driver_df = pd.DataFrame(list(drivers.items()), columns=['Factor', 'Churn Rate %'])
driver_df = driver_df.sort_values('Churn Rate %', ascending=True)
fig5 = px.bar(driver_df, x='Churn Rate %', y='Factor', orientation='h',
              color='Churn Rate %', color_continuous_scale='Reds',
              text='Churn Rate %')
fig5.update_traces(texttemplate='%{text}%', textposition='outside')
fig5.update_layout(coloraxis_showscale=False)
st.plotly_chart(fig5, use_container_width=True)