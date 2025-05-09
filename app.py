import streamlit as st 
import numpy as np 
import pandas as pd 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# App title and description
st.markdown("<h1 style='text-align: center; color: blue;'>DIADETECT</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: white;'>...a diabetes detection system</h4><br>", unsafe_allow_html=True)
st.write("Diabetes is a chronic disease that occurs when your blood glucose is too high. This application helps to effectively detect if someone has diabetes using Machine Learning.")

# Load and clean data
df = pd.read_csv("diabetes.csv")
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# Impute missing values based on Outcome
df.loc[(df['Outcome'] == 0 ) & (df['Glucose'].isnull()), 'Glucose'] = 110.6
df.loc[(df['Outcome'] == 1 ) & (df['Glucose'].isnull()), 'Glucose'] = 142.3
df.loc[(df['Outcome'] == 0 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 70.9
df.loc[(df['Outcome'] == 1 ) & (df['BloodPressure'].isnull()), 'BloodPressure'] = 75.3
df.loc[(df['Outcome'] == 0 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 27.2
df.loc[(df['Outcome'] == 1 ) & (df['SkinThickness'].isnull()), 'SkinThickness'] = 33.0
df.loc[(df['Outcome'] == 0 ) & (df['Insulin'].isnull()), 'Insulin'] = 130.3
df.loc[(df['Outcome'] == 1 ) & (df['Insulin'].isnull()), 'Insulin'] = 206.8
df.loc[(df['Outcome'] == 0 ) & (df['BMI'].isnull()), 'BMI'] = 30.9
df.loc[(df['Outcome'] == 1 ) & (df['BMI'].isnull()), 'BMI'] = 35.4

# Split features and target
X = df.drop(columns='Outcome')
y = df['Outcome']

# Scale features
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)

# Train model on selected features
selected_features = ['Insulin','Glucose','BMI','Age','SkinThickness']
x_tr = x_train[selected_features]

# User input
name = st.text_input('What is your name?').capitalize()

def get_user_input():
    insulin = st.number_input('Enter your insulin 2-Hour serum in mu U/ml')
    glucose = st.number_input('What is your plasma glucose concentration?')
    BMI = st.number_input('What is your Body Mass Index?')
    age = st.number_input('Enter your age')
    skin_thickness = st.number_input('Enter your skin fold thickness in mm')

    user_data = {
        'Insulin': insulin,
        'Glucose': glucose,
        'BMI': BMI,
        'Age': age,
        'SkinThickness': skin_thickness
    }
    return pd.DataFrame([user_data])

user_input = get_user_input()

# Predict button
if st.button('Get Result'):
    # Scale user input using the same scaler
    user_input_scaled = scaler.transform(pd.DataFrame([{
        'Pregnancies': 0,  # dummy values for unused features
        'Glucose': user_input['Glucose'][0],
        'BloodPressure': 70,
        'SkinThickness': user_input['SkinThickness'][0],
        'Insulin': user_input['Insulin'][0],
        'BMI': user_input['BMI'][0],
        'DiabetesPedigreeFunction': 0.5,
        'Age': user_input['Age'][0]
    }]))
    user_input_df = pd.DataFrame(user_input_scaled, columns=X.columns)[selected_features]

    # Train and predict
    gb = GradientBoostingClassifier(random_state=1)
    gb.fit(x_tr, y_train)
    prediction = gb.predict(user_input_df)

    # Output
    if prediction[0] == 1:
        st.error(f"{name}, you either have diabetes or are likely to have it. Please visit the doctor as soon as possible.")
    else:
        st.success(f"Hurray! {name}, you are diabetes FREE.")
