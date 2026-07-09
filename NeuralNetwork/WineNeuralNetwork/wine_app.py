import streamlit as st
import tensorflow as tf
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = tf.keras.models.load_model(
    BASE_DIR / "wine_quality_model.h5"
)

st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷"
)

st.title("🍷 Wine Quality Prediction")

st.write("Enter the wine properties below to predict its quality.")

fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.70)
citric_acid = st.number_input("Citric Acid", value=0.00)
residual_sugar = st.number_input("Residual Sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
density = st.number_input("Density", value=0.9978)
pH = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.4)

if st.button("Predict"):

    input_data = pd.DataFrame({
        "fixed acidity": [fixed_acidity],
        "volatile acidity": [volatile_acidity],
        "citric acid": [citric_acid],
        "residual sugar": [residual_sugar],
        "chlorides": [chlorides],
        "free sulfur dioxide": [free_sulfur_dioxide],
        "total sulfur dioxide": [total_sulfur_dioxide],
        "density": [density],
        "pH": [pH],
        "sulphates": [sulphates],
        "alcohol": [alcohol]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Wine Quality: {prediction[0]}")
