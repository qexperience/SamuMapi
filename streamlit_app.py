import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import requests

# Load the trained model
try:
    model = load_model("fixed_model.keras")  # Ensure the model file is correctly named and in the same directory
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit UI Elements
st.title("RGB to pH Predictor")

# Input fields for RGB values
r = st.number_input("Red (R) value (0-255):", min_value=0, max_value=255)
g = st.number_input("Green (G) value (0-255):", min_value=0, max_value=255)
b = st.number_input("Blue (B) value (0-255):", min_value=0, max_value=255)

# Predict button
if st.button("Predict pH"):
    if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
        st.error("RGB values must be between 0 and 255.")
    else:
        # Prepare the RGB data for prediction
        input_rgb = np.array([[r, g, b]])

        # Predict pH value using the loaded model
        try:
            predicted_ph = model.predict(input_rgb)[0][0]
            predicted_ph = max(0, min(14, predicted_ph))  # Clamp the pH value to the range [0, 14]
            st.success(f"Predicted pH: {round(predicted_ph, 4)}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
