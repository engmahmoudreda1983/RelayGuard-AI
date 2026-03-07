import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
    st.write("This AI-powered system predicts the failure probability of electrical protection relays by analyzing environmental and operational stressors.")
    
    st.subheader("2. Risk Equation")
    st.latex(r"z = \beta_0 + (\beta_1 \cdot A) + (\beta_2 \cdot T) + (\beta_3 \cdot S) + (\beta_4 \cdot Trip) + (\beta_5 \cdot M)")
    
    with st.expander("Show Variable Definitions"):
        st.write("**A:** Age in Months (Cumulative wear)")
        st.write("**T:** Operating Temperature (°C)")
        st.write("**S:** Voltage Spikes (Transients > 1.2x Nominal Voltage)")
        st.write("**Trip:** Tripping Time (ms)")
        st.write("**M:** Months Since Maintenance")

    st.markdown("---")
    # إخلاء المسؤولية المحدث حسب طلبك
    st.warning("⚠️ **DISCLAIMER:** This AI tool is a Decision-Support System and is intended for educational and research purposes. It should not be the sole basis for critical infrastructure decisions. All maintenance actions must be verified by a qualified Electrical Engineer using physical testing (Secondary Injection Tests) and in accordance with official utility safety protocols.")

# --- MAIN PAGE: Dashboard ---
st.title("⚡ Power Station: AI Predictive Maintenance")

col_input, col_results = st.columns([1, 2])

with col_input:
    st.header("⚙️ Sensor Inputs")
    age = st.slider("Age (Months)", 12, 240, 48)
    temp = st.slider("Operating Temperature (°C)", 20.0, 80.0, 45.0)
    spikes = st.slider("Voltage Spikes (>1.2x Un)", 0, 15, 3) # تحديث التسمية هنا أيضاً
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
            domain = {'x': [0, 1], 'y': [0.2, 1]}, 
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "black"},
                     'steps': [{'range': [0, 30], 'color': "#2ecc71"},
                               {'range': [30, 65], 'color': "#f1c40f"},
                               {'range': [65, 100], 'color': "#e74c3c"}] }
        ))
        fig_gauge.add_annotation(x=0.5, y=0.0, text=f"<b>{gauge_text}</b>", font=dict(size=26, color=gauge_color), showarrow=False)
        fig_gauge.update_layout(height=400, margin=dict(l=20, r=20, t=60, b=100))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with chart_col2:
        # رسمة الرادار
        categories = ['Age', 'Temperature', 'Spikes', 'Tripping Time', 'Maintenance']
        values = [(age/240)*100, (temp/80)*100, (spikes/15)*100, (trip_time/80)*100, (maint_months/60)*100]
        values.append(values[0]); categories.append(categories[0])
        
        fig_radar = go.Figure(go.Scatterpolar(r=values, theta=categories, fill='toself', line_color=gauge_color))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), 
                                showlegend=False, title={'text': "Sensor Stress Profile (%)", 'x': 0.5},
                                height=400, margin=dict(l=60, r=60, t=60, b=80))
        st.plotly_chart(fig_radar, use_container_width=True)