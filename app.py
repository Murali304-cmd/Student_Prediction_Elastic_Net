import streamlit as st
import pickle
import numpy as np

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# ------------------------------
# Load Model and Scaler
# ------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ------------------------------
# Title
# ------------------------------
st.title("🎓 Student Performance Prediction")
st.write("### Predict Student Performance using Elastic Net Regression")

st.markdown("---")

# ------------------------------
# User Inputs
# ------------------------------
st.subheader("Enter Student Details")

hours = st.number_input(
    "Hours Studied",
    min_value=1,
    max_value=12,
    value=6,
    step=1
)

previous_scores = st.number_input(
    "Previous Scores",
    min_value=40,
    max_value=100,
    value=75,
    step=1
)

extracurricular = st.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=4,
    max_value=10,
    value=7,
    step=1
)

sample_papers = st.number_input(
    "Sample Question Papers Practiced",
    min_value=0,
    max_value=10,
    value=5,
    step=1
)

extra = 1 if extracurricular == "Yes" else 0

st.markdown("---")

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Performance"):

    input_data = np.array([[
        hours,
        previous_scores,
        extra,
        sleep_hours,
        sample_papers
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    score = prediction[0]

    st.success(f"Predicted Performance Index: {score:.2f}")

    st.progress(min(int(score), 100))

    if score >= 90:
        st.balloons()
        st.success("⭐⭐⭐⭐⭐ Excellent Performance")

    elif score >= 75:
        st.success("⭐⭐⭐⭐ Very Good Performance")

    elif score >= 60:
        st.warning("⭐⭐⭐ Good Performance")

    else:
        st.error("⭐⭐ Needs Improvement")

st.markdown("---")
st.caption("Developed using Streamlit | Elastic Net Regression")