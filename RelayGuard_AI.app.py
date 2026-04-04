# ==============================================================================
# © 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All rights reserved.
# Standalone RelayGuard AI Edition - English Version
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime
import os

# 1. Page Config
st.set_page_config(page_title="RelayGuard AI", page_icon="🛡️", layout="wide")

# Custom CSS for the Enterprise look
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 20px; border-radius: 10px; border-left: 8px solid #9B59B6; color: white; margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Authentication STRICTLY from Secrets
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# This will only look in Streamlit Secrets Dashboard
try:
    VALID_USER = st.secrets["admin_user"]
    VALID_PASS = st.secrets["admin_pass"]
except Exception:
    st.error("🚨 CRITICAL: Credentials missing in Streamlit Secrets (admin_user / admin_pass).")
    st.stop()

if not st.session_state['logged_in']:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<div class='main-header'><h2 style='text-align: center;'>🛡️ RelayGuard AI Login</h2></div>", unsafe_allow_html=True)
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Login", use_container_width=True):
                if user == VALID_USER and pwd == VALID_PASS:
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")
    st.stop()

# 3. Main Application Logic
if 'fmea_log' not in st.session_state:
    st.session_state['fmea_log'] = pd.DataFrame(columns=['Timestamp', 'Asset ID', 'Risk %', 'Status', 'Action'])

if 'latest_eval' not in st.session_state:
    st.session_state['latest_eval'] = None

# Helper for Gauges
def create_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={'suffix': "%", 'font': {'color': color}},
        title={'text': title},
        gauge={'axis': {'range':}, 'bar': {'color': color}}))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# UI Layout
col_h, col_l = st.columns()
with col_h:
    st.markdown("<div class='main-header' style='padding: 10px;'><h2>🛡️ RelayGuard AI Assessment</h2></div>", unsafe_allow_html=True)
with col_l:
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

tab1, tab2, tab3 = st.tabs(["🛡️ Dashboard", "📑 FMEA History", "📄 AI Report"])

with tab1:
    c_in, c_dash = st.columns([1, 2.5])
    with c_in:
        with st.form("input"):
            asset = st.text_input("Asset ID", "RELAY-01")
            noise = st.number_input("Noise Index", 0.05, help="Signal fluctuations.")
            fault = st.number_input("Max Current (A)", 1500)
            age = st.number_input("Age (Months)", 192)
            alarms = st.number_input("Internal Alarms", 0)
            submit = st.form_submit_button("Evaluate", use_container_width=True)

    with c_dash:
        if submit:
            if not os.path.exists('Final_Relay_AI_Model.pkl'):
                st.error("Model file not found!")
            else:
                model = joblib.load('Final_Relay_AI_Model.pkl')
                features = np.array([[noise, fault, age, alarms]])
                
                # The Fix for your TypeError:
                pred_idx = int(np.ravel(model.predict(features)))
                probs = model.predict_proba(features)
                risk = (1 - probs) * 100
                
                classes = ['Safe Condition', 'Analog Card Issue', 'Mechanical Wear', 'CPU/Power Error']
                mode = classes[pred_idx]
                
                color = "#2ecc71" if risk < 30 else "#f1c40f" if risk < 70 else "#e74c3c"
                status = "SAFE" if risk < 30 else "WARNING" if risk < 70 else "DANGER"

                st.markdown(f"<div style='background:{color}20; border:2px solid {color}; padding:15px; border-radius:10px; text-align:center;'><h2 style='color:{color};'>{status}: {mode}</h2></div>", unsafe_allow_html=True)
                
                # Log FMEA
                new_row = pd.DataFrame([{'Timestamp': datetime.now().strftime("%H:%M:%S"), 'Asset ID': asset, 'Risk %': round(risk, 1), 'Status': status, 'Action': f"Monitor {mode}"}])
                st.session_state.fmea_log = pd.concat([new_row, st.session_state.fmea_log], ignore_index=True)
                st.session_state.latest_eval = {'id': asset, 'risk': risk, 'mode': mode, 'color': color, 'status': status, 'probs': probs, 'classes': classes}

                g1, g2 = st.columns(2)
                with g1: st.plotly_chart(create_gauge(risk, "Failure Risk", color), use_container_width=True)
                with g2:
                    fig_p = px.bar(x=probs*100, y=classes, orientation='h', color=classes, color_discrete_sequence=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'])
                    fig_p.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_p, use_container_width=True)

with tab2:
    st.dataframe(st.session_state.fmea_log, use_container_width=True)

with tab3:
    if st.session_state.latest_eval:
        ev = st.session_state.latest_eval
        st.write(f"### AI Executive Report: {ev['id']}")
        st.write(f"Risk Score: {ev['risk']:.1f}%")
        st.write(f"Identified Failure Mode: {ev['mode']}")
        st.download_button("Download Text Report", f"Asset: {ev['id']}\nRisk: {ev['risk']}%", f"Report_{ev['id']}.txt")