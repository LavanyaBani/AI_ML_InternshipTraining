import streamlit as st
from pathlib import Path
import joblib
BASE_DIR = Path(__file__).resolve().parent

st.title("BP Prediction")
Age = st.number_input("Enter your age")
weight = st.number_input("Enter your weight")
if st.button("Predict BP"):
    
    model = joblib.load(BASE_DIR /"bp_model.pkl")
    prediction = model.predict([[Age, weight]])
    st.write(f"Predicted BP: {prediction[0]}")