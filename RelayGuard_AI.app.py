# ==============================================================================
# © 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All rights reserved.
# Proprietary and Confidential. Standalone RelayGuard AI Edition.
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from datetime import datetime
import os
import warnings

# Suppress Version/Unpickling Warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. Page Configuration & Custom Styling
# ==========================================
st.set_page_config(page_title="RelayGuard AI", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    .metric-card {
        background-color: #F8F9FA; padding: 15px; border-radius: 8px; border-left: 5px solid #9B59B6; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; margin-bottom: 15px;
    }
    .main-header {
        background: linear-gradient(90deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 20px; border-radius: 10px; border-left: 8px solid #9B59B6; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Login System (STRICTLY FROM SECRETS)
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'fmea_log' not in st.session_state:
    st.session_state['fmea_log'] = pd.DataFrame(columns=['Timestamp', 'Asset ID', 'Noise Index', 'Fault Current (A)', 'Overall Risk (%)', 'Status', 'RPN Level', 'Action Required'])

if 'latest_eval' not in st.session_state:
    st.session_state['latest_eval'] = None

# Access credentials ONLY from Streamlit Secrets Dashboard
try:
    VALID_USER = st.secrets["admin_user"]
    VALID_PASS = st.secrets["admin_pass"]
except KeyError:
    st.error("🚨 SECURITY ERROR: admin_user or admin_pass NOT found in Secrets. Access Denied.")
    st.stop()

if not st.session_state['logged_in']:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<div class='main-header'><h2 style='text-align: center; margin:0;'>🛡️ RelayGuard AI Login</h2><p style='text-align: center; margin:0; color:#aaa;'>Protection Analytics Portal</p></div>", unsafe_allow_html=True)
        with st.form("login_form"):
            st.warning("🔒 Restricted Access. Authorized Personnel Only.")
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Secure Login", use_container_width=True):
                if user == VALID_USER and pwd == VALID_PASS:
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid Username or Password")
    st.stop()

# ==========================================
# 3. Helper Functions & Visualization
# ==========================================
def highlight_fmea(row):
    status = row.get('Status', '')
    if 'SAFE' in status: return ['background-color: #E8F5E9; color: #00A65A; font-weight: bold'] * len(row)
    if 'WARNING' in status: return ['background-color: #FEF9E7; color: #D35400; font-weight: bold'] * len(row)
    if 'DANGER' in status: return ['background-color: #FDEDEC; color: #C0392B; font-weight: bold'] * len(row)
    return [''] * len(row)

def create_gauge(value, title, color):
    # FIXED SYNTAX ERROR: Range value added
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={'suffix': "%", 'font': {'size': 30, 'color': color}},
        title={'text': title, 'font': {'size': 18, 'color': '#333'}},
        gauge={
            'axis': {'range':, 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'steps': [
                {'range':, 'color': 'rgba(46, 204, 113, 0.2)'}, 
                {'range':, 'color': 'rgba(241, 196, 15, 0.2)'}, 
                {'range':, 'color': 'rgba(231, 76, 60, 0.2)'}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': value}
        }))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ==========================================
# 4. Main Interface Logic
# ==========================================
col_title, col_logout = st.columns()
with col_title:
    st.markdown("<div class='main-header' style='padding: 15px;'><h2 style='margin:0;'>🛡️ RelayGuard AI - Health Assessment Dashboard</h2></div>", unsafe_allow_html=True)
with col_logout:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Logout 🚪", use_container_width=True):
        st.session_state['logged_in'] = False
        st.rerun()

tab1, tab2, tab3 = st.tabs(["🛡️ Live Assessment", "📑 FMEA & Risk Logs", "📄 AI Executive Report"])

with tab1:
    col_in, col_dash = st.columns([1, 2.5])
    
    with col_in:
        with st.form("relay_input_form"):
            st.markdown("### 📥 Telemetry Data")
            asset_id = st.text_input("Asset ID / Station Name", value="SUB-RELAY-UNIT-01")
            
            noise_index = st.number_input(
                "Pre-Fault Noise Index", value=0.05, step=0.01,
                help="Signal fluctuations before the fault event. High noise (>1.5) indicates analog card issues."
            )
            fault_current = st.number_input(
                "Max Fault Current (A)", value=1500, step=100,
                help="Maximum recorded current during the fault event."
            )
            age_months = st.number_input(
                "Relay Age (Months)", value=192, step=12,
                help="Operating age of the relay. >120 months indicates wear-out phase."
            )
            alarms_count = st.number_input(
                "Internal Alarms Count", value=0, step=1,
                help="Count of internal warnings or self-test failure logs."
            )
            
            run_btn = st.form_submit_button("🔍 Evaluate Health", use_container_width=True, type="primary")

    with col_dash:
        if run_btn:
            try:
                # Load Model
                model = joblib.load('Final_Relay_AI_Model.pkl')
                
                # Prepare Input
                features = np.array([[noise_index, fault_current, age_months, alarms_count]])
                
                # Prediction with Integer Safety Fix
                prediction_raw = model.predict(features)
                prediction_idx = int(np.ravel(prediction_raw))
                
                # Probability & Risk Calculation
                probs = model.predict_proba(features)
                risk_score = (1 - probs) * 100
                
                classes = ['Safe Condition', 'Analog Card Issue', 'Mechanical Wear', 'CPU/Power Error']
                identified_mode = classes[prediction_idx]
                
                # Risk Logic (FMEA Mapping)
                if risk_score < 30:
                    status, color, rpn, action = "🟢 SAFE", "#2ecc71", "Low", "No immediate action. Maintain TBM schedule."
                elif risk_score < 70:
                    status, color, rpn, action = "🟡 WARNING", "#f1c40f", "Medium", f"Schedule CBM inspection focusing on {identified_mode}."
                else:
                    status, color, rpn, action = "🔴 DANGER", "#e74c3c", "Critical", f"Immediate replacement recommended: {identified_mode}."

                # Update Logs
                new_entry = pd.DataFrame([{
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Asset ID': asset_id,
                    'Noise Index': noise_index,
                    'Fault Current (A)': fault_current,
                    'Overall Risk (%)': round(risk_score, 1),
                    'Status': status,
                    'RPN Level': rpn,
                    'Action Required': action
                }])
                st.session_state['fmea_log'] = pd.concat([new_entry, st.session_state['fmea_log']], ignore_index=True).head(50)
                st.session_state['latest_eval'] = {'id': asset_id, 'risk': risk_score, 'status': status, 'color': color, 'mode': identified_mode, 'action': action, 'rpn': rpn, 'probs': probs}

                # Dashboard Display
                st.markdown(f"<div style='border-radius:10px; background:{color}20; border:2px solid {color}; padding:15px; text-align:center;'><h2 style='color:{color}; margin:0;'>{status}: {identified_mode}</h2></div>", unsafe_allow_html=True)
                
                v_col1, v_col2 = st.columns(2)
                with v_col1:
                    st.plotly_chart(create_gauge(risk_score, "Total Failure Risk Score", color), use_container_width=True)
                with v_col2:
                    st.markdown("#### 📈 Probability Analysis")
                    fig_p = px.bar(x=probs*100, y=classes, orientation='h', color=classes, color_discrete_sequence=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'])
                    fig_p.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_p, use_container_width=True)

            except Exception as e:
                st.error(f"Analysis failed. Please ensure 'Final_Relay_AI_Model.pkl' is uploaded. Error: {str(e)}")

with tab2:
    st.markdown("### 📑 RPN & FMEA History Registry")
    if not st.session_state['fmea_log'].empty:
        st.dataframe(st.session_state['fmea_log'].style.apply(highlight_fmea, axis=1), use_container_width=True, hide_index=True)
    else:
        st.info("No logs available. Run an assessment to populate the database.")

with tab3:
    if st.session_state['latest_eval']:
        ev = st.session_state['latest_eval']
        st.markdown(f"### 📄 AI Executive Report - {ev['id']}")
        st.markdown(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            st.info(f"**Health Index:** {100-ev['risk']:.1f}%")
            st.warning(f"**Risk Severity (RPN):** {ev['rpn']}")
        with rep_col2:
            st.error(f"**Identified Issue:** {ev['mode']}")
            st.success(f"**Corrective Action:** {ev['action']}")
            
        report_txt = f"RELAYGUARD AI DIAGNOSTIC REPORT\nAsset: {ev['id']}\nFailure Risk: {ev['risk']:.1f}%\nStatus: {ev['status']}\nPrimary Failure Mode: {ev['mode']}\nRecommendation: {ev['action']}"
        st.download_button("📥 Download Summary Report (TXT)", data=report_txt, file_name=f"RelayGuard_Report_{ev['id']}.txt", use_container_width=True)
    else:
        st.info("💡 Run an evaluation to generate a report.")