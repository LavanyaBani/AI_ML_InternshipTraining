import streamlit as st
from pathlib import Path
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
st.title("House Rent Prediction")
import joblib
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR /"house_rent_prediction.pkl")

BHK = st.number_input("Enter the number of BHK:", min_value=1, max_value=10, step=1)
Size = st.number_input("Enter the size of the house in square feet:", min_value=100, max_value=10000, step=10)
Area_Type = st.selectbox("Select the area type:", ["Super Area", "Carpet Area", "Built Area"])
City = st.selectbox("Select the city:", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"])
Furnishing_Status = st.selectbox("Select the furnishing status:", ["Furnished", "Semi-Furnished", "Unfurnished"])
Tenant_Preferred = st.selectbox("Select the tenant preferred:", ["Family", "Bachelors", "Any"])
Bathroom = st.number_input("Enter the number of bathrooms:", min_value=1, max_value=10, step=1)
Point_of_Contact = st.selectbox("Select the point of contact:", ["Contact Owner", "Contact Agent", "Contact Builder"])
input=pd.DataFrame({
    'BHK':[BHK],
     'Size': [Size], 
     'Area Type': [Area_Type], 
     'City': [City], 
     'Furnishing Status': [Furnishing_Status],
     'Tenant Preferred': [Tenant_Preferred],
        'Bathroom': [Bathroom],
        'Point of Contact': [Point_of_Contact]
})
e=model["encoder"]
s=model["scaler"]
m=model["model"]
new_encoded = e.transform(input)
new_scaled = s.transform(new_encoded)

if st.button("Predict Rent"):
    prediction = m.predict(new_scaled)
    st.write(f"The predicted rent is: {prediction[0]}")