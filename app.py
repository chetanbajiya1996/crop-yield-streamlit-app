
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Crop Yield Prediction System")

st.markdown("""
This application predicts **crop yield** based on  
📍 **Location**, 🌱 **Crop Type**, and 📅 **Year**  
using a trained **Machine Learning model**.
""")

st.markdown("---")

st.sidebar.header("🔧 Input Parameters")

district = st.sidebar.text_input("📍 District", "Amreli")

crop = st.sidebar.selectbox(
    "🌱 Crop Type",
    ["Wheat", "Rice", "Maize", "Cotton"]
)

year = st.sidebar.number_input(
    "📅 Crop Year",
    min_value=2000,
    max_value=2050,
    value=2021
)

st.sidebar.markdown("---")
st.sidebar.info("Model uses average climate\nvalues internally.")


st.subheader("📊 Yield Prediction Result")

if st.button("🚀 Predict Yield"):
    with st.spinner("Running model prediction..."):
        df = pd.DataFrame([{
            "State_Name": "Gujarat",
            "District_Name": district,
            "Crop": crop,
            "Crop_Year": year,
            "Temperature": 28.0,
            "Humidity": 65.0,
            "Soil_Moisture": 20.0,
            "Area": 3.0
        }])

        pred_log = model.predict(df)
        predicted_yield = np.expm1(pred_log)[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("🌾 Crop", crop)
    col2.metric("📍 District", district)
    col3.metric("📦 Predicted Yield", f"{predicted_yield:.2f}")

    st.success("✅ Prediction completed successfully")

st.markdown("---")
st.subheader("ℹ️ Model Information")

st.markdown("""
- **Algorithm:** Extra Trees Regressor  
- **Target Variable:** Crop Production (Yield)  
- **Features Used:**  
  - State  
  - District  
  - Crop Type  
  - Crop Year  
  - Temperature  
  - Humidity  
  - Soil Moisture  
  - Area  

- **Deployment:** Streamlit Cloud  
- **Model Hosting:** Hugging Face Hub
""")

st.caption("© Crop Yield Prediction | ML Deployment Project")


def local_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f5f7fa;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# -----------------------------------
# Load model from Hugging Face
# -----------------------------------
MODEL_URL = "https://huggingface.co/chetanbajiya/crop-yield-model/resolve/main/yield_model.pkl"
MODEL_PATH = "/tmp/yield_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Downloading model (one-time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------------
# Streamlit UI
# -----------------------------------
st.set_page_config(page_title="Crop Yield Prediction")

st.title("🌾 Crop Yield Prediction App")
st.write("Enter District, Crop Type, and Year to predict yield")

district = st.text_input("District", "Amreli")
crop = st.selectbox("Crop", ["Wheat", "Rice", "Maize", "Cotton"])
year = st.number_input("Year", 2000, 2050, 2021)

# Default values used during training
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

    st.success(f"✅ Predicted Crop Yield: {result:.2f} units")
