import streamlit as st
st.title("House Rent Prediction")
import joblib
import pandas as pd


BHK = st.number_input("Enter BHK", min_value=1, max_value=10, step=1)
Size = st.number_input("Enter Size in sq ft", min_value=100, max_value=10000, step=10)
Area_Type = st.selectbox("Select Area Type", ["Super Area", "Carpet Area", "Built Area"])
City = st.selectbox("Select City", ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Kolkata"])
Furnishing_Status = st.selectbox("Select Furnishing Status", ["Unfurnished",    "Semi-Furnished", "Furnished"])    
tenant_preferred = st.selectbox("Select Tenant Preferred", ["Family", "Bachelors", "Any"])
Bathroom = st.number_input("Enter number of Bathrooms", min_value=1, max_value=10, step=1)
Point_of_Contact = st.selectbox("Select Point of Contact", ["Contact Owner", "Contact Agent", "Contact Builder"])   
input_data = pd.DataFrame({
    "BHK": [BHK],
    "Size": [Size],
    "Area Type": [Area_Type],
    "City": [City],
    "Furnishing Status": [Furnishing_Status],
    "Tenant Preferred": [tenant_preferred],
    "Bathroom": [Bathroom],
    "Point of Contact": [Point_of_Contact]
})
if st.button("Predict Rent"):
    model = joblib.load("house_rent_prediction.pkl")
    prediction = model.predict(input_data)
    st.write(f"Predicted Rent: {prediction[0]:,.0f}")
