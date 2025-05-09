import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and LDA transformer
model = joblib.load("diabetes_model.joblib")
lda = joblib.load("lda_transformer.joblib")  # Make sure this file exists

st.set_page_config(page_title="Diabetes Health Indicators", layout="centered")
st.title("🩺 Diabetes Risk Classifier")

st.markdown("""
<style>
    .main { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
    h1 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

st.markdown("This app predicts whether an individual is at risk of diabetes based on health indicators.")

# --- Upload Section ---
st.header("📂 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📄 Uploaded Data Preview:", df.head())

    try:
        input_data = df.values.astype(float)
        input_reduced = lda.transform(input_data)
        predictions = model.predict(input_reduced)

        df["Prediction"] = ["Anomaly" if p == -1 else "Normal" for p in predictions]
        st.success("✅ Predictions completed!")
        st.dataframe(df)

        # Option to download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# --- Manual Input Section ---
st.header("📝 Or Enter Data Manually")

features = {
    "GenHlth": st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3),
    "BMI": st.number_input("BMI", 10.0, 60.0, 25.0),
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

input_array = np.array([list(features.values())]).astype(float)

# Predict
if st.button("Predict"):
    input_reduced = lda.transform(input_data)  # Apply LDA transformation
    prediction = model.predict(input_reduced)  # Predict with SVM
    if prediction[0] == -1:
        st.error("⚠️ The model predicts: **Diabetes or Prediabetes**")
    else:
        st.success("✅ The model predicts: **No Diabetes**")
