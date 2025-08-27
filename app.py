import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Load pipeline (preprocessing + model)
model = joblib.load("model/xgb_pipeline.pkl")

# App title
st.title("📱 Mobile Price Classification App")

# Sidebar input
st.sidebar.header("Enter Mobile Specifications")

# Example input fields (you can expand this based on your dataset features)
battery_power = st.sidebar.number_input("Battery Power (mAh)", min_value=500, max_value=5000, value=2000)
ram = st.sidebar.number_input("RAM (MB)", min_value=256, max_value=8192, value=4096)
px_height = st.sidebar.number_input("Pixel Height", min_value=0, max_value=2000, value=1000)
px_width = st.sidebar.number_input("Pixel Width", min_value=0, max_value=2000, value=800)
int_memory = st.sidebar.number_input("Internal Memory (GB)", min_value=2, max_value=512, value=64)

# Collect input into dataframe
input_data = pd.DataFrame([{
    "battery_power": battery_power,
    "ram": ram,
    "px_height": px_height,
    "px_width": px_width,
    "int_memory": int_memory
}])

# Prediction
if st.button("Predict Price Category"):
    prediction = model.predict(input_data)[0]
    st.success(f"📊 Predicted Price Category: **{prediction}**")
