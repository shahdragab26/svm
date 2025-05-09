import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("Diabetes Risk Prediction Tool")
st.markdown("""
This app predicts whether a person is at risk of having diabetes based on various health indicators.
Enter your health information below to get a prediction.
""")

st.sidebar.header("User Health Information")

def user_input_features():
    highbp = st.sidebar.selectbox('High Blood Pressure', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    highchol = st.sidebar.selectbox('High Cholesterol', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    cholcheck = st.sidebar.selectbox('Cholesterol Check in Past 5 Years', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0, 0.1)
    smoker = st.sidebar.selectbox('Smoker (100+ cigarettes in lifetime)', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    stroke = st.sidebar.selectbox('Had a Stroke', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    heart_disease = st.sidebar.selectbox('Heart Disease or Attack', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    phys_activity = st.sidebar.selectbox('Physical Activity in Past 30 Days', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    fruits = st.sidebar.selectbox('Fruit Consumption (≥1 per day)', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    veggies = st.sidebar.selectbox('Vegetable Consumption (≥1 per day)', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    hvy_alcohol = st.sidebar.selectbox('Heavy Alcohol Consumption', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    healthcare = st.sidebar.selectbox('Any Healthcare Coverage', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    no_doc_cost = st.sidebar.selectbox('Could Not See Doctor Due to Cost', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    gen_health = st.sidebar.slider('General Health (1=Excellent, 5=Poor)', 1, 5, 3, 1)
    ment_health = st.sidebar.slider('Days of Poor Mental Health (Past 30 Days)', 0, 30, 0, 1)
    phys_health = st.sidebar.slider('Days of Poor Physical Health (Past 30 Days)', 0, 30, 0, 1)
    diff_walk = st.sidebar.selectbox('Difficulty Walking or Climbing Stairs', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    sex = st.sidebar.selectbox('Sex', [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    age_category = st.sidebar.slider('Age Category (1=18-24, 13=80+)', 1, 13, 6, 1)
    education = st.sidebar.slider('Education Level (1=None, 6=College Graduate)', 1, 6, 4, 1)
    income = st.sidebar.slider('Income Category (1=<$10k, 8=$75k+)', 1, 8, 4, 1)

    data = {
        'HighBP': highbp,
        'HighChol': highchol,
        'CholCheck': cholcheck,
        'BMI': bmi,
        'Smoker': smoker,
        'Stroke': stroke,
        'HeartDiseaseorAttack': heart_disease,
        'PhysActivity': phys_activity,
        'Fruits': fruits,
        'Veggies': veggies,
        'HvyAlcoholConsump': hvy_alcohol,
        'AnyHealthcare': healthcare,
        'NoDocbcCost': no_doc_cost,
        'GenHlth': gen_health,
        'MentHlth': ment_health,
        'PhysHlth': phys_health,
        'DiffWalk': diff_walk,
        'Sex': sex,
        'Age': age_category,
        'Education': education,
        'Income': income
    }

    return pd.DataFrame(data, index=[0])

# LDA weights from your notebook
lda_weights = {
    'GenHlth': 0.506704,
    'BMI': 0.383638,
    'HighBP': 0.324274,
    'Age': 0.304588,
    'HighChol': 0.235590,
    'CholCheck': 0.108139,
    'HvyAlcoholConsump': -0.105150,
    'Income': -0.099983,
    'Sex': 0.096460,
    'HeartDiseaseorAttack': 0.078778,
    'PhysHlth': -0.067320,
    'DiffWalk': 0.043655,
    'Stroke': 0.030914,
    'MentHlth': -0.028046,
    'Education': -0.023513,
    'PhysActivity': -0.016400,
    'Veggies': -0.015622,
    'AnyHealthcare': 0.008118,
    'Fruits': -0.005445,
    'Smoker': -0.004337,
    'NoDocbcCost': -0.002343
}

def compute_ld1(input_df):
    ld1 = sum(input_df[col].iloc[0] * lda_weights[col] for col in lda_weights)
    return pd.DataFrame({'LD1': [ld1]})

@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.joblib")

def predict(model, input_df):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    return prediction, probability

try:
    user_input = user_input_features()
    st.subheader("User Input Parameters")
    st.write(user_input)

    ld1_input = compute_ld1(user_input)
    model = load_model()
    prediction, probability = predict(model, ld1_input)

    st.subheader("Prediction")
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes!")
        st.write(f"Risk: {round(probability[0][1] * 100, 2)}%")
    else:
        st.success("✅ Low Risk of Diabetes")
        st.write(f"Risk: {round(probability[0][0] * 100, 2)}%")

except Exception as e:
    st.error("An error occurred.")
    st.write(f"Error details: {e}")
