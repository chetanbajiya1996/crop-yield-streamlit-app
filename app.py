import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request

# -----------------------------------------------------
# Load Model From Hugging Face
# -----------------------------------------------------
MODEL_URL = "https://huggingface.co/chetanbajiya/crop-yield-model/resolve/main/yield_model.pkl"
MODEL_PATH = "/tmp/yield_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Downloading machine learning model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="Crop Yield Prediction Dashboard",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------------------------------
# Custom CSS Styling
# -----------------------------------------------------
st.markdown("""
<style>
.title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: #2c3e50;
}
.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.result-card {
    background: linear-gradient(135deg, #4caf50, #2e7d32);
    color: white;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 5px 20px rgba(0,0,0,0.2);
}
.result-value {
    font-size: 48px;
    font-weight: 900;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# HEADER
# -----------------------------------------------------
st.markdown('<div class="title">🌾 Crop Yield Prediction Dashboard</div>', unsafe_allow_html=True)
st.write("### Machine Learning powered Smart Agriculture Tool")

# -----------------------------------------------------
# SIDEBAR INPUTS
# -----------------------------------------------------
st.sidebar.header("🔧 Input Parameters")

state = st.sidebar.text_input("🏛 State Name", "Gujarat")
district = st.sidebar.text_input("📍 District Name", "Amreli")
crop = st.sidebar.selectbox("🌱 Crop Type", ["Wheat", "Rice", "Maize", "Cotton"])
year = st.sidebar.number_input("📅 Crop Year", 2000, 2050, 2021)
temperature = st.sidebar.number_input("🌡 Temperature (°C)", 10.0, 50.0, 28.0)
humidity = st.sidebar.number_input("💧 Humidity (%)", 10.0, 100.0, 65.0)
soil_moisture = st.sidebar.number_input("🌍 Soil Moisture (%)", 1.0, 60.0, 20.0)
area = st.sidebar.number_input("📏 Cultivation Area (ha)", 0.1, 50.0, 3.0)

# ✔ NEW: UNIT SELECTION
yield_unit = st.sidebar.selectbox(
    "📦 Yield Unit",
    ["kg/ha", "ton/ha"]
)

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🚀 Predict Crop Yield", use_container_width=True)

# -----------------------------------------------------
# PREDICTION
# -----------------------------------------------------
if predict_button:
    with st.spinner("Running prediction..."):

        df = pd.DataFrame([{
            "State_Name": state,
            "District_Name": district,
            "Crop": crop,
            "Crop_Year": year,
            "Temperature": temperature,
            "Humidity": humidity,
            "Soil_Moisture": soil_moisture,
            "Area": area
        }])

        pred_log = model.predict(df)
        result_kg = np.expm1(pred_log)[0]  # base prediction in kg/ha

        # ✔ Convert units
        if yield_unit == "kg/ha":
            final_yield = result_kg
        else:
            final_yield = result_kg / 1000  # kg → ton

    # -----------------------------------------------------
    # DISPLAY RESULT
    # -----------------------------------------------------
    st.markdown("### 📦 Prediction Output")
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
            <h4>📅 Year</h4>
            <h2>{year}</h2>
        </div>
        """, unsafe_allow_html=True)

    # MAIN RESULT CARD
    st.markdown(f"""
    <div class="result-card">
        <h2>🌾 Predicted Yield</h2>
        <div class="result-value">{final_yield:.2f} {yield_unit}</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------
# FOOTER
# -----------------------------------------------------
st.markdown("""
---
### ℹ️ Model Information
**Algorithm:** Extra Trees Regressor  
**Training:** 8 Feature ML Pipeline  
**Hosting:** Hugging Face + Streamlit Cloud  
**Developer:** Crop Yield Prediction System  
""")

