import streamlit as st
from keras.models import load_model
import numpy as np
import pickle

# Load the model and scaler
model = load_model('diabetes_prediction_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit interface
st.title("Diabetes Prediction App")

st.write("""
Enter the values for the following features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
""")

# Collect user inputs
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
inputs = []

for feature in features:
    inputs.append(st.number_input(f"{feature}", value=0.0))

if st.button("Predict"):
    # Convert inputs to array
    input_data = np.array(inputs).reshape(1, -1)
    
    # Normalize the input using the scaler
    normalized_data = scaler.transform(input_data)
    
    # Predict using the model
    prediction = (model.predict(normalized_data) > 0.5).astype("int32")
    result = "Diabetes" if prediction[0][0] == 1 else "No Diabetes"
    
    st.success(f"Prediction: {result}")
