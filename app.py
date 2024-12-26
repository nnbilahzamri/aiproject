import streamlit as st
from keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('best_tuned_model.h5')

# Streamlit app title
st.title("Diabetes Prediction App")

# Subtitle
st.write("""
This app predicts the likelihood of diabetes based on user-provided health metrics.
Enter the values for the following 8 features:
""")

# Collect user inputs for the 8 features
features = {
    "Pregnancies": 0,
    "Glucose": 0,
    "Blood Pressure": 0,
    "Skin Thickness": 0,
    "Insulin": 0,
    "BMI": 0.0,
    "Diabetes Pedigree Function": 0.0,
    "Age": 0
}

inputs = []
for feature, default_value in features.items():
    if isinstance(default_value, int):
        value = st.number_input(f"{feature}", value=default_value, step=1, format="%d")
    else:
        value = st.number_input(f"{feature}", value=default_value, format="%.2f")
    inputs.append(value)

# Predict button
if st.button("Predict"):
    # Prepare input data
    input_data = np.array(inputs).reshape(1, -1)
    prediction = (model.predict(input_data) > 0.5).astype("int32")
    result = "Diabetes" if prediction[0][0] == 1 else "No Diabetes"
    
    # Display the prediction result
    st.success(f"Prediction: {result}")
