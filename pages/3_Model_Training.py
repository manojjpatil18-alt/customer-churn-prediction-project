import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import plotly.express as px

# =========================================
# PAGE TITLE
# =========================================

st.title("🤖 Machine Learning Model Training")

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("customer_churn.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

# =========================================
# REMOVE DUPLICATES
# =========================================

df = df.drop_duplicates()

# =========================================
# HANDLE CATEGORICAL DATA
# =========================================

encoder = LabelEncoder()

categorical_columns = [
    "Gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "InternetService",
    "Churn"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# =========================================
# FEATURES & TARGET
# =========================================

X = df.drop("Churn", axis=1)
y = df["Churn"]

# =========================================
# FEATURE SCALING
# =========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# MODELS
# =========================================

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True)
}

# =========================================
# TRAINING
# =========================================

results = []

progress = st.progress(0)

for i, (name, model) in enumerate(models.items()):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": round(accuracy * 100, 2)
    })

    progress.progress((i + 1) * 20)

# =========================================
# RESULTS
# =========================================

results_df = pd.DataFrame(results)

st.subheader("📊 Model Comparison")

st.dataframe(results_df)

# =========================================
# ACCURACY CHART
# =========================================

fig = px.bar(
    results_df,
    x="Model",
    y="Accuracy",
    color="Model",
    title="Model Accuracy Comparison"
)

st.plotly_chart(fig, use_container_width=True)

# =========================================
# BEST MODEL
# =========================================

best_model = RandomForestClassifier()

best_model.fit(X_train, y_train)

# =========================================
# SAVE MODEL
# =========================================

pickle.dump(best_model, open("saved_model.pkl", "wb"))

pickle.dump(scaler, open("scaler.pkl", "wb"))

# =========================================
# PREDICTION
# =========================================

y_pred = best_model.predict(X_test)

# =========================================
# CLASSIFICATION REPORT
# =========================================

st.subheader("📄 Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)

# =========================================
# CONFUSION MATRIX
# =========================================

st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

st.write(cm)

# =========================================
# FEATURE IMPORTANCE
# =========================================

importance = best_model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

fig2 = px.bar(
    feature_df,
    x="Feature",
    y="Importance",
    color="Feature",
    title="Feature Importance"
)

st.plotly_chart(fig2, use_container_width=True)

st.success("✅ Model Trained and Saved Successfully")