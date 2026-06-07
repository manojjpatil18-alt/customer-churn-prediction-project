import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

df = pd.read_csv("customer_churn.csv")

total_customers = len(df)
churn_customers = len(df[df['Churn'] == 'Yes'])
retention_rate = ((total_customers - churn_customers) / total_customers) * 100
avg_charges = df['MonthlyCharges'].mean()

st.title("📈 Business Intelligence Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Total Customers", total_customers)
col2.metric("❌ Churn Customers", churn_customers)
col3.metric("✅ Retention Rate", f"{retention_rate:.2f}%")
col4.metric("💰 Avg Charges", f"${avg_charges:.2f}")

st.markdown("---")

# Pie Chart
fig1 = px.pie(
    df,
    names='Churn',
    title='Customer Churn Distribution'
)

st.plotly_chart(fig1, use_container_width=True)

# Bar Chart
fig2 = px.bar(
    df,
    x='InternetService',
    color='Churn',
    title='Internet Service vs Churn'
)

st.plotly_chart(fig2, use_container_width=True)

st.info("""
### 📌 Business Problem

Telecom companies lose customers frequently.
This system helps predict churn customers early.
""")