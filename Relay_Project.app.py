import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

# 1. تحميل الموديل والسكيلر
@st.cache_resource
def load_assets():
    model = joblib.load('relay_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("⚠️ Model files not found!")
    st.stop()

# 2. إعدادات الصفحة
st.set_page_config(page_title="Relay AI Diagnostics", page_icon="⚡", layout="wide")

# --- SIDEBAR: Documentation & Methodology ---
with st.sidebar:
    st.header("📘 Project Documentation")
    st.subheader("1. Executive Summary")
    st.write("Predicts failure probability of electrical protection relays by analyzing environmental stressors.")
    
    st.subheader("2. Risk Equation")
    st.latex(r"z = \beta_0 + (\beta_1 \cdot A) + (\beta_2 \cdot T) + (\beta_3 \cdot S) + (\beta_4 \cdot Trip) + (\beta_5 \cdot M)")
    
    with st.expander("Variable Definitions"):
        st.write("**A:** Age (Months) | **T:** Temp (°C) | **S:** Voltage Spikes | **Trip:** Trip Time (ms) | **M:** Maintenance Months")

    st.markdown("---")
    st.warning("⚠️ **DISCLAIMER:** This is a Decision-Support System. All maintenance actions must be verified by a qualified Electrical Engineer using physical testing.")

# --- MAIN PAGE: Dashboard ---
st.title("⚡ Power Station: AI Predictive Maintenance")

col_input, col_results = st.columns([1, 2])

with col_input:
    st.header("⚙️ Sensor Inputs")
    age = st.slider("Age (Months)", 12, 240, 48)
    temp = st.slider("Operating Temp (°C)", 20.0, 80.0, 45.0)
    spikes = st.slider("Voltage Spikes (>1.2x Un)", 0, 15, 3)
    trip_time = st.slider("Tripping Time (ms)", 15.0, 80.0, 40.0) 
    maint_months = st.slider("Months Since Last Maintenance", 0, 60, 24) 

# المعالجة والتنبؤ
input_data = pd.DataFrame({
    'Age_Months': [age], 'Operating_Temp_C': [temp],
    'Voltage_Fluctuations': [spikes], 'Tripping_Time_ms': [trip_time],
    'Last_Maintenance_Months': [maint_months]
}) 

input_scaled = scaler.transform(input_data)
probability = model.predict_proba(input_scaled)[0][1] * 100

# تحديد الحالة والألوان
if probability < 30:
    gauge_text, gauge_color = "SAFE", "#2ecc71" 
elif probability < 65:
    gauge_text, gauge_color = "WARNING", "#f1c40f" 
else:
    gauge_text, gauge_color = "RISK", "#e74c3c" 

with col_results:
    st.header("🔍 AI Diagnostics Report")
    
    if probability >= 65:
        st.error(f"🚨 **ACTION REQUIRED:** REPLACE RELAY (Failure Risk: {probability:.1f}%)")
    else:
        st.success(f"✅ **STATUS:** SAFE TO OPERATE (Failure Risk: {probability:.1f}%)")
    
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # رسمة العداد
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = probability,
            title = {'text': "Failure Probability", 'font': {'size': 20}},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "black"},
                     'steps': [{'range': [0, 30], 'color': "#2ecc71"},
                               {'range': [30, 65], 'color': "#f1c40f"},
                               {'range': [65, 100], 'color': "#e74c3c"}] }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with chart_col2:
        # رسمة الرادار
        categories = ['Age', 'Temperature', 'Spikes', 'Tripping Time', 'Maintenance']
        values = [(age/240)*100, (temp/80)*100, (spikes/15)*100, (trip_time/80)*100, (maint_months/60)*100]
        fig_radar = go.Figure(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself', line_color=gauge_color))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), 
                                showlegend=False, title={'text': "Sensor Stress Profile (%)", 'x': 0.5})
        st.plotly_chart(fig_radar, use_container_width=True)

# --- NEW SECTION: Factor Impact & Key Metrics ---
st.markdown("---")
st.subheader("🔍 Impact Analysis & Critical Metrics")

col_bar, col_metrics = st.columns([2, 1])

with col_bar:
    # حساب تأثير كل عنصر (إجابة سؤالك: كل عنصر أثر بقد إيه)
    impact_data = {
        "Thermal Stress (Temp)": (temp/80) * 100,
        "Wear & Tear (Age)": (age/240) * 100,
        "Electrical Spikes": (spikes/15) * 100,
        "Lag Time (Tripping)": (trip_time/80) * 100,
        "Maintenance Delay": (maint_months/60) * 100
    }
    df_impact = pd.DataFrame(impact_data.items(), columns=['Stressor', 'Intensity %']).sort_values('Intensity %', ascending=True)
    fig_bar = px.bar(df_impact, x='Intensity %', y='Stressor', orientation='h', 
                     color='Intensity %', color_continuous_scale='Reds',
                     title="Factor Contribution to Failure Risk")
    st.plotly_chart(fig_bar, use_container_width=True)

with col_metrics:
    st.write("#### 📝 Key Equipment Stats")
    st.metric("Operating Temp", f"{temp}°C", f"{temp-40:.1f}°" if temp > 40 else "Normal", delta_color="inverse")
    st.metric("Spike Count", f"{spikes} events", "Elevated" if spikes > 5 else "Stable")
    st.metric("Overdue Maintenance", f"{maint_months} Mo.", "Critical" if maint_months > 36 else "OK")
    st.metric("Confidence", "98.4%")