import streamlit as st
import pandas as pd
import joblib

# Load saved model and encoder
model = joblib.load("xgb_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Streamlit app title
st.title("📱 Mobile Price Category Predictor")

st.write("Enter mobile specifications below to predict the price category:")

# Input fields
battery_power = st.number_input("Battery Power (mAh)", min_value=500, max_value=5000, step=100)
ram = st.number_input("RAM (MB)", min_value=256, max_value=8192, step=128)
px_height = st.number_input("Pixel Resolution Height", min_value=0, max_value=2000, step=10)
px_width = st.number_input("Pixel Resolution Width", min_value=0, max_value=2000, step=10)
mobile_wt = st.number_input("Mobile Weight (grams)", min_value=50, max_value=400, step=1)
n_cores = st.number_input("Number of Cores", min_value=1, max_value=8, step=1)
int_memory = st.number_input("Internal Memory (GB)", min_value=2, max_value=128, step=1)
talk_time = st.number_input("Talk Time (hours)", min_value=2, max_value=20, step=1)

# Create dataframe from input
input_data = pd.DataFrame([{
    "battery_power": battery_power,
    "ram": ram,
    "px_height": px_height,
    "px_width": px_width,
    "mobile_wt": mobile_wt,
    "n_cores": n_cores,
    "int_memory": int_memory,
    "talk_time": talk_time
}])

# Predict button
if st.button("🔮 Predict Price Category"):
    prediction = model.predict(input_data)
    category = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted Price Category: **{category}**")
