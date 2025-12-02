import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

MODEL_PATH = "/tmp/yield_model.pkl"   # ✅ writable on Streamlit Cloud

def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?id=1yDuyeNwzgZmgyx34qcXHJ8Cc8t4EFh4A"
        gdown.download(url, MODEL_PATH, quiet=False)
    return joblib.load(MODEL_PATH)

model = load_model()



st.set_page_config(page_title="Crop Yield Prediction")

st.title("🌾 Crop Yield Prediction App")

district = st.text_input("District", "Amreli")
crop = st.selectbox("Crop", ["Wheat", "Rice", "Maize", "Cotton"])
year = st.number_input("Year", 2000, 2050, 2021)

state_name = "Gujarat"
temperature = 28.0
humidity = 65.0
soil_moisture = 20.0
area = 3.0

if st.button("Predict Yield"):
    df = pd.DataFrame([{
        "State_Name": state_name,
        "District_Name": district,
        "Crop": crop,
        "Crop_Year": year,
        "Temperature": temperature,
        "Humidity": humidity,
        "Soil_Moisture": soil_moisture,
        "Area": area
    }])

    pred_log = model.predict(df)
    result = np.expm1(pred_log)[0]

    st.success(f"✅ Predicted Yield: {result:.2f}")
