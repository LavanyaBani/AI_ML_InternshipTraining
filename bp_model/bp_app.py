import streamlit as st
st.title("BP Prediction")
Age = st.number_input("Enter your age")
weight = st.number_input("Enter your weight")
if st.button("Predict BP"):
    import joblib
    model = joblib.load("bp_model.pkl")
    prediction = model.predict([[Age, weight]])
    st.write(f"Predicted BP: {prediction[0]}")