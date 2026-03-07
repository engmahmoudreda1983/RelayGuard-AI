import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib  # المكتبة المسئولة عن تحميل الموديل المحفوظ

# 1. تحميل الموديل والسكيلر (الملفات اللي حفظنا فيها الذكاء)
@st.cache_resource
def load_assets():
    model = joblib.load('relay_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("⚠️ ملفات الموديل (pkl) غير موجودة في نفس الفولدر!")
    st.stop()

# 2. إعدادات الصفحة
st.set_page_config(page_title="Relay AI Diagnostics", page_icon="⚡", layout="wide")
st.title("⚡ Power Station: AI Predictive Maintenance")

# 3. مدخلات المستخدم (المنطقة اليسرى)
col_input, col_results = st.columns([1, 2])

with col_input:
    st.header("⚙️ Sensor Inputs")
    age = st.slider("Age (Months)", 12, 240, 48)
    temp = st.slider("Operating Temperature (°C)", 20.0, 80.0, 45.0)
    spikes = st.slider("Voltage Fluctuations", 0, 15, 3)
    trip_time = st.slider("Tripping Time (ms)", 15.0, 80.0, 40.0) 
    maint_months = st.slider("Months Since Last Maintenance", 0, 60, 24) 

# 4. المعالجة والتنبؤ
input_data = pd.DataFrame({
    'Age_Months': [age], 
    'Operating_Temp_C': [temp],
    'Voltage_Fluctuations': [spikes], 
    'Tripping_Time_ms': [trip_time],
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

# 5. عرض النتائج (المنطقة اليمنى)
with col_results:
    st.header("🔍 AI Diagnostics Report")
    
    if probability >= 65:
        st.error(f"🚨 **ACTION REQUIRED:** REPLACE RELAY (Failure Risk: {probability:.1f}%)")
    else:
        st.success(f"✅ **STATUS:** SAFE TO OPERATE (Failure Risk: {probability:.1f}%)")
    
    st.markdown("---")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # رسمة العداد (Gauge)
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
        # رسمة الشبكة (Radar Chart)
        categories = ['Age', 'Temperature', 'Spikes', 'Tripping Time', 'Maintenance']
        values = [(age/240)*100, (temp/80)*100, (spikes/15)*100, (trip_time/80)*100, (maint_months/60)*100]
        values.append(values[0]); categories.append(categories[0])
        
        fig_radar = go.Figure(go.Scatterpolar(r=values, theta=categories, fill='toself', line_color=gauge_color))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), 
                                showlegend=False, title={'text': "Sensor Stress Profile (%)", 'x': 0.5},
                                height=400, margin=dict(l=60, r=60, t=60, b=80))
        st.plotly_chart(fig_radar, use_container_width=True)