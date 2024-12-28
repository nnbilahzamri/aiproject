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
| **BMI**                    | Body Mass Index (18.5 to 60).                      | `24.5`, `33.6`      |
| **Diabetes Pedigree Function** | Genetic risk factor (0 to 2.0).               | `0.351`, `0.627`    |
| **Age**                    | Age in years (1 to 100).                          | `25`, `45`, `60`    |

**Note:** *Pre-filled value is the minimum value for each field.*
""")

# Initialize session state for inputs
default_values = {
    'pregnancies': 0,
    'glucose': 60,
    'blood_pressure': 50,
    'skin_thickness': 10,
    'insulin': 35,
    'bmi': 18.5,
    'diabetes_pedigree_function': 0.0,
    'age': 1,
    'prediction_made': False,
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Input fields, updated to reflect the session state after the reset
pregnancies = st.text_input('Pregnancies', value=str(st.session_state['pregnancies']))
glucose = st.text_input('Glucose Level', value=str(st.session_state['glucose']))
blood_pressure = st.text_input('Blood Pressure', value=str(st.session_state['blood_pressure']))
skin_thickness = st.text_input('Skin Thickness', value=str(st.session_state['skin_thickness']))
insulin = st.text_input('Insulin Level', value=str(st.session_state['insulin']))
bmi = st.text_input('BMI', value=str(st.session_state['bmi']))
diabetes_pedigree_function = st.text_input('Diabetes Pedigree Function', value=str(st.session_state['diabetes_pedigree_function']))
age = st.text_input('Age', value=str(st.session_state['age']))


# Parse inputs and handle errors
try:
    user_input = np.array([[int(pregnancies), int(glucose), int(blood_pressure),
                            int(skin_thickness), int(insulin), float(bmi),
                            float(diabetes_pedigree_function), int(age)]])
except ValueError:
    st.warning("Please enter valid numerical values for all fields.")
    st.stop()

# Button to trigger prediction
if st.button('Predict'):
    try:
        # Scale the input using the same scaler used during training
        user_input_scaled = scaler.transform(user_input)

        # Make prediction
        prediction = model.predict(user_input_scaled)

        # Display the result
        if prediction >= 0.7:
            st.markdown(
                '<div style="background-color: #f9e79f; padding: 15px; border-radius: 10px; color: black;">'
                '<b>The model predicts: Diabetes</b>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background-color: #d5f5e3; padding: 15px; border-radius: 10px; color: black;">'
                '<b>The model predicts: No Diabetes</b>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Update session state
        st.session_state['prediction_made'] = True

    except Exception as e:
        st.error(f"Error in scaling or prediction: {str(e)}")

# Show "Predict New" button if a prediction has been made
if st.session_state['prediction_made']:
    if st.button('Predict New'):
        # Reset session state values to defaults
        for key in default_values.keys():
            st.session_state[key] = default_values[key]
        st.session_state['prediction_made'] = False
