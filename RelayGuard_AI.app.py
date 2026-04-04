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
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# 1. Page Configuration
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

# Credentials from Streamlit Secrets
try:
    VALID_USER = st.secrets["admin_user"]
    VALID_PASS = st.secrets["admin_pass"]
except KeyError:
    st.error("🚨 SECURITY ERROR: admin_user/admin_pass not found in Secrets. Please configure them in Streamlit Dashboard.")
    st.stop()

if not st.session_state['logged_in']:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<div class='main-header'><h2 style='text-align: center; margin:0;'>🛡️ RelayGuard AI Login</h2><p style='text-align: center; margin:0; color:#aaa;'>Protection Asset Analytics</p></div>", unsafe_allow_html=True)
        with st.form("login_form"):
            st.warning("🔒 Restricted Access. Authorized Personnel Only.")
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Secure Login", use_container_width=True):
                if user == VALID_USER and pwd == VALID_PASS:
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid Credentials")
    st.stop()

# ==========================================
# 3. Helper Functions
# ==========================================
def highlight_fmea(row):
    status = row.get('Status', '')
    if 'SAFE' in status: return ['background-color: #E8F5E9; color: #00A65A; font-weight: bold'] * len(row)
    if 'WARNING' in status: return ['background-color: #FEF9E7; color: #D35400; font-weight: bold'] * len(row)
    if 'DANGER' in status: return ['background-color: #FDEDEC; color: #C0392B; font-weight: bold'] * len(row)
    return [''] * len(row)

def create_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={'suffix': "%", 'font': {'size': 30, 'color': color}},
        title={'text': title, 'font': {'size': 18, 'color': '#333'}},
        gauge={
            'axis': {'range':},
            'bar': {'color': color},
            'bgcolor': "white",
            'steps': [
                {'range':, 'color': 'rgba(46, 204, 113, 0.2)'}, 
                {'range':, 'color': 'rgba(241, 196, 15, 0.2)'}, 
                {'range':, 'color': 'rgba(231, 76, 60, 0.2)'}
            ]}))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ==========================================
# 4. Main UI Logic
# ==========================================
col_h, col_l = st.columns()
with col_h:
    st.markdown("<div class='main-header' style='padding: 15px;'><h2 style='margin:0;'>🛡️ RelayGuard AI - Health Assessment</h2></div>", unsafe_allow_html=True)
with col_l:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Logout 🚪"):
        st.session_state['logged_in'] = False
        st.rerun()

tab1, tab2, tab3 = st.tabs(["🛡️ Assessment Dashboard", "📑 FMEA Logs", "📄 Executive Report"])

with tab1:
    col_in, col_res = st.columns([1, 2.2])
    with col_in:
        with st.form("input_form"):
            asset_id = st.text_input("Asset ID", value="RELAY-UNIT-01")
            noise = st.number_input("Pre-Fault Noise Index", value=0.05, help="Standard signal fluctuations.")
            current = st.number_input("Max Fault Current (A)", value=1500, help="Peak current during fault.")
            age = st.number_input("Relay Age (Months)", value=192, help="Age since installation.")
            alarms = st.number_input("Internal Alarms", value=0, help="Error logs count.")
            btn = st.form_submit_button("📊 Evaluate Health", use_container_width=True)

    with col_res:
        if btn:
            try:
                model = joblib.load('Final_Relay_AI_Model.pkl')
                # Input and Prediction
                features = np.array([[noise, current, age, alarms]])
                pred_raw = model.predict(features)
                # Fix for TypeError: extract scalar from array
                idx = int(np.ravel(pred_raw))
                
                probs = model.predict_proba(features)
                risk = (1 - probs) * 100
                classes = ['Safe Condition', 'Analog Card Issue', 'Mechanical Wear', 'CPU/Power Error']
                
                if risk < 30: status, color, rpn, action = "🟢 SAFE", "#2ecc71", "Low", "Continue standard maintenance."
                elif risk < 70: status, color, rpn, action = "🟡 WARNING", "#f1c40f", "Medium", f"Check {classes[idx]}."
                else: status, color, rpn, action = "🔴 DANGER", "#e74c3c", "Critical", f"Immediate action for {classes[idx]}."

                st.markdown(f"<div style='background:{color}20; border:2px solid {color}; border-radius:10px; padding:15px; text-align:center;'><h2 style='color:{color}; margin:0;'>{status}: {classes[idx]}</h2></div>", unsafe_allow_html=True)
                
                # FMEA Log Entry
                new_log = pd.DataFrame([{'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Asset ID': asset_id, 'Noise Index': noise, 'Fault Current (A)': current, 'Overall Risk (%)': round(risk, 1), 'Status': status, 'RPN Level': rpn, 'Action Required': action}])
                st.session_state.fmea_log = pd.concat([new_log, st.session_state.fmea_log], ignore_index=True)
                st.session_state.latest_eval = {'id': asset_id, 'risk': risk, 'status': status, 'color': color, 'mode': classes[idx], 'action': action, 'rpn': rpn, 'probs': probs}

                g1, g2 = st.columns(2)
                with g1: st.plotly_chart(create_gauge(risk, "Failure Risk Score", color), use_container_width=True)
                with g2:
                    st.markdown("#### 📈 Probability Breakdown")
                    fig_p = px.bar(x=probs*100, y=classes, orientation='h', color=classes, color_discrete_sequence=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'])
                    fig_p.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_p, use_container_width=True)

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

with tab2:
    st.markdown("### 📑 RPN & FMEA History Log")
    if not st.session_state.fmea_log.empty:
        st.dataframe(st.session_state.fmea_log.style.apply(highlight_fmea, axis=1), use_container_width=True)

with tab3:
    if st.session_state.latest_eval:
        ev = st.session_state.latest_eval
        st.markdown(f"### 📄 AI Executive Report - {ev['id']}")
        st.info(f"**Health Score:** {100-ev['risk']:.1f}% | **Failure Mode:** {ev['mode']}")
        st.success(f"**Action Plan:** {ev['action']}")
        report = f"Asset: {ev['id']}\nRisk: {ev['risk']:.1f}%\nStatus: {ev['status']}\nAction: {ev['action']}"
        st.download_button("📥 Download Report", data=report, file_name=f"Report_{ev['id']}.txt")