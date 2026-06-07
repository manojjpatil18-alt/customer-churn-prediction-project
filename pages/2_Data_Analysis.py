import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

st.title("📊 Exploratory Data Analysis")

df = pd.read_csv("customer_churn.csv")

# Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Missing Values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# Duplicate Records
st.subheader("Duplicate Records")
st.write(df.duplicated().sum())

# Histogram
st.subheader("Monthly Charges Distribution")

fig1 = px.histogram(
    df,
    x="MonthlyCharges",
    nbins=20,
    color="Churn"
)

st.plotly_chart(fig1)

# Box Plot
st.subheader("Tenure Analysis")

fig2 = px.box(
    df,
    x="Churn",
    y="Tenure",
    color="Churn"
)

st.plotly_chart(fig2)

# Pie Chart
fig3 = px.pie(
    df,
    names="Gender",
    title="Gender Distribution"
)

st.plotly_chart(fig3)

# Correlation Heatmap
st.subheader("Correlation Heatmap")

numeric_df = df.select_dtypes(include=['number'])

corr = numeric_df.corr()

fig, ax = plt.subplots()

heatmap = ax.imshow(corr)

ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns)

ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)

plt.colorbar(heatmap)

st.pyplot(fig)