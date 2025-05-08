import streamlit as st
import pandas as pd
import joblib


model = joblib.load("model_knowledge.pkl")

st.title("SVM Classifier Model")

uploaded_file = st.file_uploader("Please upload a CSV file so I can use my brain!", type="csv")

# If the user uploads a file...
if uploaded_file:
    # ...read the file
    df = pd.read_csv(uploaded_file)
    st.write("Here's a sneak peek at what you uploaded:")
    st.write(df.head())

    # ...make predictions
    predictions = model.predict(df)

    # ...show the predictions
    st.write("Here are the predictions from my brain:")
    st.write(predictions)