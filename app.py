import streamlit as st
import numpy as np
import pickle
from keras.models import load_model

# Load the trained model
model = load_model('best_tuned_model.h5')

# Load the scaler
with open('./scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streamlit UI for user input
st.title('Diabetes Prediction')

# Guidelines for data input in a table
st.markdown("""
### Guidelines for Input Data:
| **Feature**                | **Description**                                    | **Example Input**   |
|----------------------------|---------------------------------------------------|---------------------|
| **Pregnancies**            | Number of pregnancies (0 or more).                | `0`, `2`, `4`       |
| **Glucose Level**          | Plasma glucose concentration (60 to 200).          | `85`, `120`, `150`  |
| **Blood Pressure**         | Diastolic blood pressure in mmHg (50 to 200).      | `70`, `80`, `120`   |
| **Skin Thickness**         | Skin fold thickness in mm (10 to 100).             | `20`, `35`, `50`    |
| **Insulin Level**          | Serum insulin in μU/ml (35 to 450).                | `90`, `120`, `200`  |
| **BMI**                    | Body Mass Index (18.5 to 60).        | `24.5`, `33.6`      |
| **Diabetes Pedigree Function** | Genetic risk factor (0 to 2.0). | `0.351`, `0.627`    |
| **Age**                    | Age in years (1 to 100).                          | `25`, `45`, `60`    |
""")

# Create input fields for all the features
pregnancies = st.number_input('Pregnancies', min_value=0, step=1, value=0, format='%d')
glucose = st.number_input('Glucose Level', min_value=60, max_value=200, step=1, value=60)
blood_pressure = st.number_input('Blood Pressure', min_value=50, max_value=200, step=1, value=50)
skin_thickness = st.number_input('Skin Thickness', min_value=10, max_value=100, step=1, value=10)
insulin = st.number_input('Insulin Level', min_value=35, max_value=450, step=1, value=35)
bmi = st.number_input('BMI', min_value=18.5, max_value=60.0, value=18.5, step=0.1, format="%.1f")
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.0, value=0.0, step=0.001, format="%.3f")
age = st.number_input('Age', min_value=1, max_value=100, step=1, value=1)

# Collect all inputs into a list or array
user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

# Button to trigger prediction
if st.button('Predict'):
    try:
        # Scale the input using the same scaler used during training
        user_input_scaled = scaler.transform(user_input)

        # Make prediction
        prediction = model.predict(user_input_scaled)

        # Display the result
        if prediction >= 0.7:
            st.success('The model predicts: **Diabetes**')
        else:
            st.success('The model predicts: **No Diabetes**')

    except Exception as e:
        st.error(f"Error in scaling or prediction: {str(e)}")
