import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Load trained model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("yield_model.pkl")

model = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Crop Yield Prediction")

st.title("🌾 Crop Yield Prediction App")
st.write("Enter District, Crop Type, and Year to get predicted yield.")

district = st.text_input("District Name", "Amreli")
crop = st.selectbox("Crop Type", ["Wheat", "Rice", "Maize", "Cotton"])
year = st.number_input("Crop Year", 2000, 2050, 2021)

# Default values required by the model
state_name = "Gujarat"
temperature = 28.0
humidity = 65.0
soil_moisture = 20.0
area = 3.0

if st.button("Predict Yield"):
    input_df = pd.DataFrame([{
        "State_Name": state_name,
        "District_Name": district,
        "Crop": crop,
        "Crop_Year": year,
        "Temperature": temperature,
        "Humidity": humidity,
        "Soil_Moisture": soil_moisture,
        "Area": area
    }])

    pred_log = model.predict(input_df)
    predicted_yield = np.expm1(pred_log)[0]

    st.success(f"✅ Predicted Crop Yield: {predicted_yield:.2f} units")

