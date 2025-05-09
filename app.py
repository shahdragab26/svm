import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load model and scaler
model = joblib.load('diabetes_model.joblib')
scaler = joblib.load('scaler.joblib')  # Make sure you saved this earlier
lda = LinearDiscriminantAnalysis(n_components=1)

st.title("🩺 Diabetes Risk Classifier")
st.markdown("This app predicts whether an individual is at risk of diabetes based on health indicators.")

# Input fields
def user_input():
    data = {
        'HighBP': st.selectbox('High Blood Pressure', [0, 1]),
        'HighChol': st.selectbox('High Cholesterol', [0, 1]),
        'CholCheck': st.selectbox('Cholesterol Check in Past 5 Years', [0, 1]),
        'BMI': st.slider('Body Mass Index', 10.0, 100.0, 25.0),
        'Smoker': st.selectbox('Smoker', [0, 1]),
        'Stroke': st.selectbox('Stroke', [0, 1]),
        'HeartDiseaseorAttack': st.selectbox('Heart Disease or Attack', [0, 1]),
        'PhysActivity': st.selectbox('Physical Activity', [0, 1]),
        'Fruits': st.selectbox('Consumes Fruit Daily', [0, 1]),
        'Veggies': st.selectbox('Consumes Vegetables Daily', [0, 1]),
        'HvyAlcoholConsump': st.selectbox('Heavy Alcohol Consumption', [0, 1]),
        'AnyHealthcare': st.selectbox('Has Healthcare Coverage', [0, 1]),
        'NoDocbcCost': st.selectbox('Skipped Doctor Due to Cost', [0, 1]),
        'GenHlth': st.selectbox('General Health (1=Excellent, 5=Poor)', [1, 2, 3, 4, 5]),
        'MentHlth': st.slider('Poor Mental Health Days (last 30)', 0, 30, 0),
        'PhysHlth': st.slider('Poor Physical Health Days (last 30)', 0, 30, 0),
        'DiffWalk': st.selectbox('Difficulty Walking', [0, 1]),
        'Sex': st.selectbox('Sex (0=Female, 1=Male)', [0, 1]),
        'Age': st.selectbox('Age Category (1=18-24, ..., 13=80+)', list(range(1, 14))),
        'Education': st.selectbox('Education Level (1=None, 6=College Grad)', list(range(1, 7))),
        'Income': st.selectbox('Income Level (1=<10k, 8=75k+)', list(range(1, 9)))
    }
    return pd.DataFrame([data])

input_df = user_input()

# Preprocessing
input_scaled = scaler.transform(input_df)
input_lda = lda.fit_transform(input_scaled, [0])  # Dummy target for transform

# Prediction
prediction = model.predict(input_lda)

# Output
st.subheader("Prediction Result")
if prediction[0] == 1:
    st.error("⚠️ The model predicts: **Diabetes or Prediabetes**")
else:
    st.success("✅ The model predicts: **No Diabetes**")

