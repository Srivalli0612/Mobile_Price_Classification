import streamlit as st
import pandas as pd
import xgboost as xgb

# Load model
model = xgb.XGBClassifier()
model.load_model("model/xgb_model.json")

st.title("📱 Mobile Price Classification")
st.write("Enter the mobile specifications below to predict the price range.")

# Input fields for all features
battery_power = st.number_input("Battery Power (mAh)", min_value=500, max_value=6000, step=100)
blue = st.selectbox("Bluetooth", [0, 1])
clock_speed = st.number_input("Clock Speed (GHz)", min_value=0.5, max_value=5.0, step=0.1)
dual_sim = st.selectbox("Dual SIM", [0, 1])
fc = st.number_input("Front Camera (MP)", min_value=0, max_value=20, step=1)
four_g = st.selectbox("4G", [0, 1])
int_memory = st.number_input("Internal Memory (GB)", min_value=2, max_value=128, step=1)
m_dep = st.number_input("Mobile Depth (cm)", min_value=0.1, max_value=1.0, step=0.1)
mobile_wt = st.number_input("Mobile Weight (g)", min_value=80, max_value=250, step=1)
n_cores = st.number_input("Number of Cores", min_value=1, max_value=8, step=1)
pc = st.number_input("Primary Camera (MP)", min_value=0, max_value=20, step=1)
px_height = st.number_input("Pixel Height", min_value=0, max_value=2000, step=10)
px_width = st.number_input("Pixel Width", min_value=0, max_value=2000, step=10)
ram = st.number_input("RAM (MB)", min_value=256, max_value=8192, step=256)
sc_h = st.number_input("Screen Height (cm)", min_value=5, max_value=20, step=1)
sc_w = st.number_input("Screen Width (cm)", min_value=0, max_value=20, step=1)
talk_time = st.number_input("Talk Time (hours)", min_value=2, max_value=20, step=1)
three_g = st.selectbox("3G", [0, 1])
touch_screen = st.selectbox("Touch Screen", [0, 1])
wifi = st.selectbox("WiFi", [0, 1])

if st.button("Predict"):
    # Put inputs into dataframe in correct order
    input_data = pd.DataFrame([{
        "battery_power": battery_power,
        "blue": blue,
        "clock_speed": clock_speed,
        "dual_sim": dual_sim,
        "fc": fc,
        "four_g": four_g,
        "int_memory": int_memory,
        "m_dep": m_dep,
        "mobile_wt": mobile_wt,
        "n_cores": n_cores,
        "pc": pc,
        "px_height": px_height,
        "px_width": px_width,
        "ram": ram,
        "sc_h": sc_h,
        "sc_w": sc_w,
        "talk_time": talk_time,
        "three_g": three_g,
        "touch_screen": touch_screen,
        "wifi": wifi
    }])
    
    prediction = model.predict(input_data)[0]
    
    categories = {
        0: "Low Cost 💸",
        1: "Medium Cost 💰",
        2: "High Cost 📱",
        3: "Very High Cost 🚀"
    }
    
    st.success(f"Predicted Price Category: {categories[prediction]}")
