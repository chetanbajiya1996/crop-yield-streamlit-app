
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request
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

st.set_page_config(
    page_title="Crop Yield Prediction Dashboard",
    page_icon="🌾",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
.css-18e3th9 {
    padding: 2rem;
}
.css-1d391kg {
    background-color: #ffffff;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}
.title {
    font-size: 36px;
    font-weight: 700;
}
.subtitle {
    font-size: 18px;
    color: #555;
}
.result {
    font-size: 40px;
    font-weight: 700;
    color: #2E7D32;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="card">
  <div class="title">🌾 Crop Yield Prediction Dashboard</div>
  <div class="subtitle">
    Machine Learning based yield estimation using location & crop information
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


st.sidebar.header("🔧 Input Parameters")

district = st.sidebar.text_input("📍 District Name", "Amreli")

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
st.sidebar.info("ℹ️ Climate & field parameters\nare set to average values.")


st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Predict Crop Yield", use_container_width=True):
    with st.spinner("⏳ Running prediction model..."):
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

    with col1:
        st.markdown(f"""
        <div class="card">
            <h4>📍 District</h4>
            <h2>{district}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <h4>🌱 Crop</h4>
            <h2>{crop}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <h4>📦 Predicted Yield</h4>
            <div class="result">{predicted_yield:.2f}</div>
        </div>
        """, unsafe_allow_html=True)


st.markdown("<br><hr>", unsafe_allow_html=True)

st.markdown("""
**Model Details**
- Algorithm: Extra Trees Regressor  
- Deployment: Streamlit Cloud  
- Model Hosting: Hugging Face  
- Prediction Domain: Agriculture  

© Crop Yield Prediction System
""")

# -----------------------------------
# Load model from Hugging Face


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
