import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page settings
st.set_page_config(page_title="🎓 Student Exam Score Predictor", page_icon="🎯", layout="centered")
st.title("🎓 Student Exam Score Predictor")

# --- Input Fields ---
col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("📘 Study Hours per Day", 0.0, 12.0, 2.0)
    attendance = st.slider("📊 Attendance (%)", 0.0, 100.0, 80.0)
    sleep_hours = st.slider("😴 Sleep Hours per Day", 0.0, 12.0, 7.0)
    exercise_frequency = st.slider("🏃 Exercise Frequency (Days/Week)", 0, 7, 3)

with col2:
    mental_health = st.slider("🧠 Mental Health Rating (1–10)", 1, 10, 5)
    social_media_hours = st.slider("📱 Social Media Hours/Day", 0.0, 10.0, 2.0)
    netflix_hours = st.slider("🎬 Netflix/OTT Hours/Day", 0.0, 10.0, 1.0)
    part_time_job = st.selectbox("💼 Part-time Job", ["Yes", "No"])

# Encode categorical value
ptj_encoded = 1 if part_time_job == "Yes" else 0

# Predict Button
if st.button("🔍 Predict"):
    # Format input into correct shape
    input_data = np.array([[
        study_hours,
        attendance,
        sleep_hours,
        exercise_frequency,
        mental_health,
        social_media_hours,
        netflix_hours,
        ptj_encoded
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # Keep prediction between 0–100
    prediction = max(0, min(100, prediction))

    # Display result
    st.metric(label="🎯 Predicted Exam Score", value=f"{prediction:.2f}%")
    st.progress(int(prediction))

