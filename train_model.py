import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("customer_churn.csv")

print("Dataset Loaded Successfully")
print(df.head())

# =========================
# DATASET INFO
# =========================

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

# =========================
# REMOVE DUPLICATES
# =========================

df = df.drop_duplicates()

# =========================
# CHECK MISSING VALUES
# =========================

print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# LABEL ENCODING
# =========================

encoder = LabelEncoder()

categorical_columns = [
    'Gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'InternetService',
    'Churn'
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

# =========================
# FEATURES AND TARGET
# =========================

X = df.drop("Churn", axis=1)
y = df["Churn"]

# =========================
# FEATURE SCALING
# =========================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL TRAINING
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# MODEL PREDICTION
# =========================

y_pred = model.predict(X_test)

# =========================
# MODEL EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy * 100)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================

pickle.dump(model, open("saved_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("\nModel Saved Successfully")