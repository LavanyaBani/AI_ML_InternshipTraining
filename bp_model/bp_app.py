import streamlit as st
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR /"bp_model.pkl")
st.title("BP Prediction")
Age = st.number_input("Enter your age")
weight = st.number_input("Enter your weight")
if st.button("Predict BP"):
    import joblib
    model = joblib.load("bp_model.pkl")
    prediction = model.predict([[Age, weight]])
    st.write(f"Predicted BP: {prediction[0]}")