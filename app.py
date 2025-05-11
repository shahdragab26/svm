import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Diabetes Risk Classifier", layout="centered")

# Load model and transformer
lda = joblib.load("lda_transformer.joblib")
model = joblib.load("diabetes_model.joblib")

# --- Title and Image ---
st.title("🩺 Diabetes Risk Classifier")
st.image("diabetes.jpg", caption="Stay healthy. Early detection saves lives!", use_container_width=True)
st.markdown("Use this tool to check diabetes risk based on your health inputs.")

# --- Input Form ---
st.header("📋 Enter Your Health Information")

features = {
    "GenHlth": st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3),
    "BMI": st.number_input("Body Mass Index (BMI)", 10.0, 60.0, 25.0),
    "HighBP": st.selectbox("High Blood Pressure", [0, 1]),
    "Age": st.slider("Age", 18, 100, 40),
    "HighChol": st.selectbox("High Cholesterol", [0, 1]),
    "CholCheck": st.selectbox("Cholesterol Check in Last 5 Years", [0, 1]),
    "HvyAlcoholConsump": st.selectbox("Heavy Alcohol Consumption", [0, 1]),
    "Income": st.slider("Income Level (1=Low, 8=High)", 1, 8, 4),
    "Sex": st.selectbox("Sex (0=Female, 1=Male)", [0, 1]),
    "HeartDiseaseorAttack": st.selectbox("Heart Disease or Attack", [0, 1]),
    "PhysHlth": st.slider("Physical Health (days unwell in past 30)", 0, 30, 5),
    "DiffWalk": st.selectbox("Difficulty Walking", [0, 1]),
    "Stroke": st.selectbox("Ever had a Stroke", [0, 1]),
    "MentHlth": st.slider("Mental Health (days unwell in past 30)", 0, 30, 5),
    "Education": st.slider("Education Level (1=Low, 6=High)", 1, 6, 4),
    "PhysActivity": st.selectbox("Physical Activity", [0, 1]),
    "Veggies": st.selectbox("Eats Vegetables Regularly", [0, 1]),
    "AnyHealthcare": st.selectbox("Has Any Healthcare Coverage", [0, 1]),
    "Fruits": st.selectbox("Eats Fruits Regularly", [0, 1]),
    "Smoker": st.selectbox("Smoker", [0, 1]),
    "NoDocbcCost": st.selectbox("Couldn’t See Doctor Due to Cost", [0, 1]),
}

input_data = np.array([list(features.values())]).astype(float)

# --- Prediction Logic Based on Rules ---
if st.button("🔍 Predict"):
    st.subheader("🔎 Result")

    # Simple rule-based logic
    risk_score = 0

    # Add points for various risk factors
    if features["GenHlth"] >= 4:
        risk_score += 1
    if features["BMI"] >= 30:
        risk_score += 1
    if features["HighBP"] == 1:
        risk_score += 1
    if features["HighChol"] == 1:
        risk_score += 1
    if features["HeartDiseaseorAttack"] == 1:
        risk_score += 1
    if features["DiffWalk"] == 1:
        risk_score += 1
    if features["Stroke"] == 1:
        risk_score += 1
    if features["Age"] >= 45:
        risk_score += 1
    if features["PhysActivity"] == 0:
        risk_score += 1
    if features["Smoker"] == 1:
        risk_score += 1
    if features["PhysHlth"] > 10:
        risk_score += 1
    if features["MentHlth"] > 10:
        risk_score += 1

    # Define threshold (adjust as needed)
    if risk_score >= 5:
        st.error("⚠️ Based on your inputs, you may be at **risk of Diabetes**.")
        st.markdown("**Final classification: DIABETIC**")
    else:
        st.success("✅ Based on your inputs, you are **unlikely to have Diabetes**.")
        st.markdown("**Final classification: NOT DIABETIC**")

# Footer
st.markdown("---")
st.caption("Made with ❤️ for educational purposes. Not a substitute for medical advice.")
