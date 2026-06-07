import streamlit as st

st.set_page_config(
    page_title="AI Powered Customer Churn Prediction System",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class='main-header'>
    <h1>📊 AI Powered Customer Churn Prediction System</h1>
    <p>Final Year Data Science Project using Machine Learning & Streamlit</p>
</div>
""", unsafe_allow_html=True)

st.image(
    "https://cdn-icons-png.flaticon.com/512/4149/4149647.png",
    width=180
)

st.markdown("""
## 🚀 Project Overview

This system predicts whether a telecom customer is likely to leave the company using Machine Learning algorithms.

### 🌐 Domains Used
- Business Intelligence
- Exploratory Data Analysis
- Machine Learning
- Predictive Analytics
- Artificial Intelligence
- Data Visualization
- Model Evaluation
- Streamlit Deployment
""")

st.success("✅ System Loaded Successfully")