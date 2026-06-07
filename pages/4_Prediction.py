import streamlit as st
import pickle
import numpy as np

st.title("🔮 AI Customer Churn Prediction")

model = pickle.load(open("saved_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# User Inputs
Gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
Tenure = st.slider("Tenure", 1, 72)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)
MonthlyCharges = st.number_input(
    "Monthly Charges",
    20.0,
    120.0
)
TotalCharges = st.number_input(
    "Total Charges",
    0.0,
    10000.0
)

# Encoding
gender_map = {
    "Male": 1,
    "Female": 0
}

yes_no = {
    "Yes": 1,
    "No": 0
}

internet_map = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}

input_data = np.array([[
    gender_map[Gender],
    SeniorCitizen,
    yes_no[Partner],
    yes_no[Dependents],
    Tenure,
    yes_no[PhoneService],
    internet_map[InternetService],
    MonthlyCharges,
    TotalCharges
]])

scaled_data = scaler.transform(input_data)

if st.button("Predict Churn"):

    prediction = model.predict(scaled_data)

    probability = model.predict_proba(scaled_data)

    risk = probability[0][1] * 100

    if prediction[0] == 1:

        st.error(f"⚠️ Customer likely to churn ({risk:.2f}% Risk)")

        st.warning("""
        ### AI Recommendation
        - Offer discounts
        - Improve customer support
        - Provide loyalty rewards
        """)

    else:

        st.success(f"✅ Customer likely to stay ({100-risk:.2f}% Confidence)")