import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pre-trained models
lda = joblib.load('lda_model.pkl')
svm = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define your feature names (update these to match your model)
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

st.title("Diabetes Prediction using Pre-trained LDA and SVM")

st.write("### Enter Your Health Data")
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    input_lda = lda.transform(input_scaled)
    input_lda_scaled = scaler.transform(input_lda)
    prediction = svm.predict(input_lda_scaled)
    result = "Diabetes" if prediction[0] == 1 else "No Diabetes"
    st.success(f"### Prediction: {result}")

