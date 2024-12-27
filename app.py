import streamlit as st
import numpy as np
import pickle
from keras.models import load_model

# Load the trained model
model = load_model('diabetes_prediction_model.h5')

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit UI for user input
st.title('Diabetes Prediction')

# Create input fields for the user
glucose = st.number_input('Glucose Level', min_value=0, max_value=400)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100)
insulin = st.number_input('Insulin Level', min_value=0, max_value=900)
bmi = st.number_input('BMI', min_value=10, max_value=60)

# Collect all inputs into a list or array
user_input = np.array([[glucose, blood_pressure, skin_thickness, insulin, bmi]])

# Scale the input using the same scaler used during training
user_input_scaled = scaler.transform(user_input)

# Make prediction
prediction = model.predict(user_input_scaled)

# Display the result
if prediction >= 0.6:
    st.write('The model predicts: **Diabetes**')
else:
    st.write('The model predicts: **No Diabetes**')
