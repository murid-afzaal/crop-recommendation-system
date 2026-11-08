import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(page_title="Smart Crop Recommendation System", page_icon="🌾", layout="centered")

# Load Model
@st.cache_resource
def load_model():
    with open('best_crop_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

data = load_model()
model = data["model"]
scaler = data["scaler"]
label_encoder = data["label_encoder"]
model_type = data["model_type"]

st.title(" Smart Crop Recommendation System")
st.write("Provide your soil nutrient values and weather conditions below to find the most suitable crop for cultivation.")

# Input Sliders / Number Inputs
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, value=50.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=145.0, value=50.0)
    K = st.number_input("Potassium (K)", min_value=0.0, max_value=205.0, value=50.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)

# Prediction Button
if st.button("🔍 Recommend Crop", use_container_width=True):
    user_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    if model_type in ["Logistic Regression", "K-Nearest Neighbors (KNN)", "Support Vector Machine (SVM)"]:
        user_input = scaler.transform(user_input)
        
    prediction_encoded = model.predict(user_input)
    crop_name = label_encoder.inverse_transform(prediction_encoded)[0].upper()
    
    st.success(f" The most suitable crop to grow is: **{crop_name}**")