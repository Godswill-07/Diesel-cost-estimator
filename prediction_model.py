import streamlit as st
import pandas as pd
import joblib
import json


#load trained model
gen_runtime_model = joblib.load('gen_runtime_model_best.pkl')

st.title("Generator Diesel Cost Estimator")

st.image("generator.png", caption="Diesel Generator", use_column_width=True)

st.header("Please enter required inputs")

load_kw = st.number_input("Load(KW)", min_value=0.0,max_value=5.0,value=3.0)
phcn_uptime = st.number_input("PHCN Uptime (Hours)", min_value=0.0, max_value=12.0,value=6.0)
solar_output = st.number_input("Solar Output (Kwh)", min_value=0.0,max_value=10.0,value=4.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0,max_value=45.0,value=0.0)
battery_voltage = st.number_input("Battery Voltgage (V)", min_value=0.0,max_value=14.0,value=12.0)
power_source = st.selectbox("Power source", ['Gen', 'PHCN', 'Solar', 'Hybrid'])
diesel_price = st.number_input("Diesel Price (₦/liter)", value=1018.75)

st.markdown('---')

#input for runtime prediction
input_data_gen = pd.DataFrame({
    'power_source':[power_source],
    'load_kw':[load_kw],
    'phcn_uptime': [phcn_uptime],
    'solar_output': [solar_output],
    'temperature': [temperature]
})

#predicted generator runtime
predicted_gen_hours = gen_runtime_model.predict(input_data_gen)[0]

st.subheader("Generator Runtime Prediction")
st.write(f"**Estimated Runtime: {predicted_gen_hours:.2f} hours**")

average_diesel_consumption_perday = 3 #litres per day
prediected_diesel_cost = predicted_gen_hours * average_diesel_consumption_perday * diesel_price

st.subheader('Estimated diesel cost')
st.write(f"₦{prediected_diesel_cost:,.2f} per day ")

with open('model_metrics.json_1', 'r') as f:
    metrics = json.load(f)

st.sidebar.header("Model Performance (Gen Runtime)")
st.sidebar.write(f"**Train MSE:** {metrics['train_mse']:.2f}")
st.sidebar.write(f"**Test MSE:** {metrics['test_mse']:.2f}")
st.sidebar.write(f"**Train RMSE:** {metrics['train_rmse']:.2f}")
st.sidebar.write(f"**Test RMSE:** {metrics['test_rmse']:.2f}")


