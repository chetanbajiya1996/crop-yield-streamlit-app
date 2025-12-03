import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import urllib.request

# =============================================================
# 1️⃣ LOAD MODEL FROM HUGGING FACE
# =============================================================
MODEL_URL = "https://huggingface.co/chetanbajiya/crop-yield-model/resolve/main/yield_model.pkl"
MODEL_PATH = "/tmp/yield_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("⬇️ Downloading Machine Learning Model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return joblib.load(MODEL_PATH)

model = load_model()


# =============================================================
# 2️⃣ STREAMLIT PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="Crop Yield Prediction Dashboard",
    page_icon="🌾",
    layout="wide"
)


# =============================================================
# 3️⃣ CUSTOM CSS (PREMIUM UI)
# =============================================================
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
}
.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}
.result-card {
    background: linear-gradient(135deg, #4caf50, #2e7d32);
    color: white;
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 5px 20px rgba(0,0,0,0.2);
}
.result-value {
    font-size: 50px;
    font-weight: 900;
}
</style>
""", unsafe_allow_html=True)


# =============================================================
# 4️⃣ HEADER
# =============================================================
st.markdown("""
<h1 style='text-align:center; color:#2c3e50;'>
🌾 Advanced Crop Yield Prediction Dashboard
</h1>
<p style='text-align:center; font-size:18px;'>Smart ML-powered yield estimation</p>
""", unsafe_allow_html=True)


# =============================================================
# 5️⃣ SIDEBAR INPUTS
# =============================================================
st.sidebar.header("🔧 Input Parameters")

state = st.sidebar.text_input("🏛 State Name", "Gujarat")
district = st.sidebar.text_input("📍 District Name", "Amreli")
crop = st.sidebar.selectbox("🌱 Crop Type", ["Wheat", "Rice", "Maize", "Cotton"])
year = st.sidebar.number_input("📅 Crop Year", 2000, 2050, 2021)

temperature = st.sidebar.number_input("🌡 Temperature (°C)", 10.0, 50.0, 28.0)
humidity = st.sidebar.number_input("💧 Humidity (%)", 10.0, 100.0, 65.0)
soil_moisture = st.sidebar.number_input("🌍 Soil Moisture (%)", 1.0, 60.0, 20.0)
area = st.sidebar.number_input("📏 Cultivation Area (ha)", 0.1, 50.0, 3.0)

predict_button = st.sidebar.button("🚀 Predict Crop Yield", use_container_width=True)


# =============================================================
# 6️⃣ SHOW INPUT METRICS
# =============================================================
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🌡 Temperature", f"{temperature} °C")
with col2:
    st.metric("💧 Humidity", f"{humidity} %")
with col3:
    st.metric("🌍 Soil Moisture", f"{soil_moisture} %")

st.markdown("---")


# =============================================================
# 7️⃣ PREDICTION
# =============================================================
if predict_button:
    with st.spinner("🔍 Running prediction..."):

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

        # ✅ MODEL OUTPUT ASSUMED IN kg/ha
        yield_kg_per_ha = model.predict(df)[0]

        # ✅ safety
        yield_kg_per_ha = round(max(yield_kg_per_ha, 0), 1)


    # =============================================================
    # 8️⃣ DISPLAY RESULT
    # =============================================================
    st.markdown("## 📊 Yield Prediction Results")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="card">
            <h4>📍 District</h4>
            <h2>{district}</h2>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <h4>🌱 Crop</h4>
            <h2>{crop}</h2>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <h4>📅 Year</h4>
            <h2>{year}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <h2>🌾 Predicted Yield</h2>
        <div class="result-value">{yield_kg_per_ha}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================
# 9️⃣ FOOTER
# =============================================================
st.markdown("""
---
### ℹ️ Model Information
- **Algorithm:** Extra Trees Regressor  
- **Prediction Target:** Yield  
- **Hosting:** Hugging Face + Streamlit  
- **Developer:** Crop Yield Prediction System  
""")

