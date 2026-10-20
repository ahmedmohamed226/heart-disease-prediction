import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Heart Disease Predictor")

st.title("Heart Disease Prediction")
st.write("Enter the patient details and get a prediction.")
st.caption("Learning project only, not medical advice.")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load():
    model = joblib.load(os.path.join(BASE, "model_artifacts", "heart_model.pkl"))
    scaler = joblib.load(os.path.join(BASE, "model_artifacts", "scaler.pkl"))
    features = joblib.load(os.path.join(BASE, "model_artifacts", "features.pkl"))
    return model, scaler, features

try:
    model, scaler, features = load()
except FileNotFoundError:
    st.error("Couldn't find the model files. Run the notebook first to generate model_artifacts.")
    st.stop()

st.sidebar.header("Patient details")
age = st.sidebar.number_input("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex (1 = male, 0 = female)", [1, 0])
cp = st.sidebar.selectbox("Chest pain type (0-3)", [0, 1, 2, 3], index=2)
trestbps = st.sidebar.number_input("Resting blood pressure", 90, 200, 130)
chol = st.sidebar.number_input("Cholesterol", 100, 600, 240)
fbs = st.sidebar.selectbox("Fasting blood sugar > 120 (1 = yes, 0 = no)", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.sidebar.number_input("Max heart rate", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise induced angina (1 = yes, 0 = no)", [0, 1])
oldpeak = st.sidebar.number_input("ST depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)
slope = st.sidebar.selectbox("Slope of ST segment (0-2)", [0, 1, 2], index=1)
ca = st.sidebar.selectbox("Number of major vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thal (1 = normal, 2 = fixed defect, 3 = reversible defect)", [1, 2, 3], index=2)

if st.button("Predict"):
    row = {f: 0 for f in features}
    values = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }
    for k, v in values.items():
        if k in row:
            row[k] = v

    X = np.array([[row[f] for f in features]])
    pred = model.predict(scaler.transform(X))[0]

    if pred == 1:
        st.error("Prediction: heart disease detected")
    else:
        st.success("Prediction: no heart disease")
