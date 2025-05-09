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

# App title and introduction
st.title("Diabetes Risk Prediction Tool")
st.markdown("""
This app predicts whether a person is at risk of having diabetes based on various health indicators.
Enter your health information below to get a prediction.
""")

# Create sidebar for inputs
st.sidebar.header("User Health Information")

# Functions to collect user inputs
def user_input_features():
    # Create input fields for each feature in your model
    highbp = st.sidebar.selectbox('High Blood Pressure', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    highchol = st.sidebar.selectbox('High Cholesterol', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    cholcheck = st.sidebar.selectbox('Cholesterol Check in Past 5 Years', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0, 0.1)
    smoker = st.sidebar.selectbox('Smoker (100+ cigarettes in lifetime)', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    stroke = st.sidebar.selectbox('Had a Stroke', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    heart_disease = st.sidebar.selectbox('Heart Disease or Attack', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    phys_activity = st.sidebar.selectbox('Physical Activity in Past 30 Days', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    fruits = st.sidebar.selectbox('Fruit Consumption (≥1 per day)', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    veggies = st.sidebar.selectbox('Vegetable Consumption (≥1 per day)', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    hvy_alcohol = st.sidebar.selectbox('Heavy Alcohol Consumption', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    healthcare = st.sidebar.selectbox('Any Healthcare Coverage', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    no_doc_cost = st.sidebar.selectbox('Could Not See Doctor Due to Cost', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    gen_health = st.sidebar.slider('General Health (1=Excellent, 5=Poor)', 1, 5, 3, 1)
    ment_health = st.sidebar.slider('Days of Poor Mental Health (Past 30 Days)', 0, 30, 0, 1)
    phys_health = st.sidebar.slider('Days of Poor Physical Health (Past 30 Days)', 0, 30, 0, 1)
    diff_walk = st.sidebar.selectbox('Difficulty Walking or Climbing Stairs', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    sex = st.sidebar.selectbox('Sex', options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    age_category = st.sidebar.slider('Age Category (1=18-24, 13=80+)', 1, 13, 6, 1)
    education = st.sidebar.slider('Education Level (1=None, 6=College Graduate)', 1, 6, 4, 1)
    income = st.sidebar.slider('Income Category (1=<$10k, 8=$75k+)', 1, 8, 4, 1)
    
    # Organize inputs into a DataFrame
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
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
user_input = user_input_features()

# Display the user inputs
st.subheader('User Input Parameters')
st.write(user_input)

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load('./diabetes_model.joblib')
    return model

# Make prediction with the model
def predict(model, input_df):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    return prediction, probability

# Make prediction when model is available
try:
    model = load_model()
    prediction, probability = predict(model, user_input)
    
    # Display the prediction
    st.subheader('Prediction')
    diabetes_prediction = prediction[0]
    if diabetes_prediction == 1:
        st.error('⚠️ High Risk of Diabetes!')
        risk_percentage = round(probability[0][1] * 100, 2)
        st.write(f"The model predicts a {risk_percentage}% chance of diabetes or prediabetes.")
    else:
        st.success('✅ Low Risk of Diabetes')
        no_risk_percentage = round(probability[0][0] * 100, 2)
        st.write(f"The model predicts a {no_risk_percentage}% chance of not having diabetes.")
    
    # Display additional information
    st.subheader('Risk Factors')
    risk_factors = []
    if user_input['HighBP'].iloc[0] == 1:
        risk_factors.append("High Blood Pressure")
    if user_input['HighChol'].iloc[0] == 1:
        risk_factors.append("High Cholesterol")
    if user_input['BMI'].iloc[0] > 30:
        risk_factors.append("Obesity (BMI > 30)")
    if user_input['Smoker'].iloc[0] == 1:
        risk_factors.append("Smoker")
        
    if risk_factors:
        st.write("Your key risk factors include:")
        for factor in risk_factors:
            st.write(f"- {factor}")
    else:
        st.write("No major risk factors identified!")
        
except:
    st.error("Please make sure the model file 'diabetes_model.joblib' is uploaded.")
