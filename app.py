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

# Create input fields for all the features
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20)
glucose = st.number_input('Glucose Level', min_value=0, max_value=300)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100)
insulin = st.number_input('Insulin Level', min_value=0, max_value=900)
bmi = st.number_input('BMI', min_value=10, max_value=60)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5)
age = st.number_input('Age', min_value=1, max_value=120)

# Collect all inputs into a list or array
user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

# Ensure no input is empty or zero (if you want to enforce validation)
if np.any(user_input == 0):
    st.warning("Please provide non-zero values for all inputs.")

else:
    # Scale the input using the same scaler used during training
    try:
        user_input_scaled = scaler.transform(user_input)

        # Make prediction
        prediction = model.predict(user_input_scaled)

        # Display the result
        if prediction >= 0.5:
            st.write('The model predicts: **Diabetes**')
        else:
            st.write('The model predicts: **No Diabetes**')

    except Exception as e:
        st.error(f"Error in scaling or prediction: {str(e)}")
