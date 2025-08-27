import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import joblib

# Load preprocessing pipeline (without classifier)
preprocessor = joblib.load("model/preprocessor.pkl")

# Load XGBoost model
clf = XGBClassifier()
clf.load_model("model/xgb_model.json")

st.title("📱 Mobile Price Category Predictor")

# Inputs
battery_power = st.slider("Battery Power", 500, 2000, 1000)
ram = st.slider("RAM (MB)", 256, 4000, 1500)
px_height = st.slider("Pixel Height", 0, 1960, 600)
px_width = st.slider("Pixel Width", 0, 2000, 1000)
int_memory = st.slider("Internal Memory (GB)", 2, 128, 32)
mobile_wt = st.slider("Mobile Weight (g)", 80, 250, 150)
n_cores = st.slider("No. of Cores", 1, 8, 4)
four_g = st.selectbox("4G Support", [0, 1])
touch_screen = st.selectbox("Touch Screen", [0, 1])

if st.button("Predict"):
    sample = pd.DataFrame([{
        'battery_power': battery_power,
        'blue': 1,
        'clock_speed': 1.5,
        'dual_sim': 1,
        'fc': 1,
        'four_g': four_g,
        'int_memory': int_memory,
        'm_dep': 0.5,
        'mobile_wt': mobile_wt,
        'n_cores': n_cores,
        'pc': 2,
        'px_height': px_height,
        'px_width': px_width,
        'ram': ram,
        'sc_h': 10,
        'sc_w': 5,
        'talk_time': 10,
        'three_g': 1,
        'touch_screen': touch_screen,
        'wifi': 1
    }])

    # Preprocess (if you had a preprocessor)
    X = preprocessor.transform(sample)

    # Predict
    prediction = clf.predict(X)
    label_map = {0: "Low", 1: "Mid", 2: "High", 3: "Premium"}
    st.success(f"Predicted Price Category: {label_map[prediction[0]]}")
