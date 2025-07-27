# popularity_app.py

import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Nepali Song Popularity Predictor", layout="centered")

# Load model and scaler
model_path = "./models/Ridge_popularity_model.pkl"
scaler_path = "./models/scaler.pkl"

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"Failed to load model or scaler.\n\n**Details:** {e}")
    st.stop()

st.title("🎵 Nepali Song Popularity Predictor")
st.markdown("Enter the audio features of a Nepali song to predict its **popularity (0–100)**.")

# Feature inputs
features = {
    'BPM': st.slider('🎚️ BPM (Tempo)', 60, 200, 120),
    'Energy': st.slider('⚡ Energy', 0, 100, 50),
    'Dance': st.slider('💃 Danceability', 0, 100, 50),
    'Loud': st.slider('🔊 Loudness', 0, 100, 50),
    'Valence': st.slider('😊 Valence (Mood)', 0, 100, 50),
    'Length': st.slider('⏱️ Length (seconds)', 60, 900, 240),
    'Acoustic': st.slider('🎸 Acousticness', 0, 100, 50)
}

# Prepare input
input_data = np.array([[features[key] for key in features]])
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Popularity"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"**Predicted Popularity: {prediction:.1f} / 100**")
