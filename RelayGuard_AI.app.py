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
    cols_fmea = list()
    cols_fmea.append('Timestamp')
    cols_fmea.append('Asset ID')
    cols_fmea.append('Noise Index')
    cols_fmea.append('Max Current (A)')
    cols_fmea.append('Risk (%)')
    cols_fmea.append('Status')
    cols_fmea.append('Action')
    st.session_state['fmea_log'] = pd.DataFrame(columns=cols_fmea)

if 'latest_eval' not in st.session_state:
    st.session_state['latest_eval'] = None

# STRICT SECURITY: Access credentials ONLY from Streamlit Secrets Dashboard
try:
    VALID_USER = st.secrets["admin_user"]
    VALID_PASS = st.secrets["admin_pass"]
except Exception:
    st.error("🚨 SECURITY ERROR: admin_user or admin_pass NOT found in Secrets Dashboard. Access Denied.")
    st.stop()

if not st.session_state['logged_in']:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    col_sizes1 = list()
    col_sizes1.append(1)
    col_sizes1.append(1.5)
    col_sizes1.append(1)
    col1, col2, col3 = st.columns(col_sizes1)
    
    with col2:
        st.markdown("<div class='main-header'><h2 style='text-align: center; margin:0;'>🛡️ RelayGuard AI Login</h2><p style='text-align: center; margin:0; color:#aaa;'>Enterprise Protection Analytics Portal</p></div>", unsafe_allow_html=True)
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
    if 'SAFE' in status:
        res = list()
        res.append('background-color: #E8F5E9; color: #00A65A; font-weight: bold')
        return res * len(row)
    if 'WARNING' in status:
        res = list()
        res.append('background-color: #FEF9E7; color: #D35400; font-weight: bold')
        return res * len(row)
    if 'DANGER' in status:
        res = list()
        res.append('background-color: #FDEDEC; color: #C0392B; font-weight: bold')
        return res * len(row)
    res = list()
    res.append('')
    return res * len(row)

def create_gauge(value, title, color):
    # BULLETPROOF GAUGE: Built piece by piece to prevent markdown deletion bugs
    r_full = list()
    r_full.append(0)
    r_full.append(100)

    r_safe = list()
    r_safe.append(0)
    r_safe.append(30)

    r_warn = list()
    r_warn.append(30)
    r_warn.append(70)

    r_danger = list()
    r_danger.append(70)
    r_danger.append(100)

    ax_dict = dict()
    ax_dict.update({"range": r_full})
    ax_dict.update({"tickwidth": 1})

    step1 = dict()
    step1.update({"range": r_safe})
    step1.update({"color": "rgba(46, 204, 113, 0.2)"})

    step2 = dict()
    step2.update({"range": r_warn})
    step2.update({"color": "rgba(241, 196, 15, 0.2)"})

    step3 = dict()
    step3.update({"range": r_danger})
    step3.update({"color": "rgba(231, 76, 60, 0.2)"})

    steps_list = list()
    steps_list.append(step1)
    steps_list.append(step2)
    steps_list.append(step3)

    thresh = dict()
    thresh.update({"line": {"color": "black", "width": 4}})
    thresh.update({"thickness": 0.75})
    thresh.update({"value": value})

    g_dict = dict()
    g_dict.update({"axis": ax_dict})
    g_dict.update({"bar": {"color": color}})
    g_dict.update({"bgcolor": "white"})
    g_dict.update({"steps": steps_list})
    g_dict.update({"threshold": thresh})

    num_dict = dict()
    num_dict.update({"suffix": "%"})
    num_dict.update({"font": {"size": 30, "color": color}})

    title_dict = dict()
    title_dict.update({"text": title})
    title_dict.update({"font": {"size": 18, "color": "#333"}})

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=num_dict,
        title=title_dict,
        gauge=g_dict
    ))
    
    layout_margin = dict()
    layout_margin.update({"l": 20})
    layout_margin.update({"r": 20})
    layout_margin.update({"t": 50})
    layout_margin.update({"b": 20})
    
    fig.update_layout(height=300, margin=layout_margin, paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ==========================================
# 4. Main Interface Logic
# ==========================================
col_sizes2 = list()
col_sizes2.append(10)
col_sizes2.append(2)
col_title, col_logout = st.columns(col_sizes2)

with col_title:
    st.markdown("<div class='main-header' style='padding: 15px;'><h2 style='margin:0;'>🛡️ RelayGuard AI - Health Dashboard</h2></div>", unsafe_allow_html=True)
with col_logout:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Logout 🚪", use_container_width=True):
        st.session_state['logged_in'] = False
        st.rerun()

tabs_names = list()
tabs_names.append("🛡️ Diagnostic Dashboard")
tabs_names.append("📑 FMEA & Risk Logs")
tabs_names.append("📄 Executive Report")
tab1, tab2, tab3 = st.tabs(tabs_names)

with tab1:
    col_sizes3 = list()
    col_sizes3.append(1)
    col_sizes3.append(2.5)
    col_in, col_dash = st.columns(col_sizes3)
    
    with col_in:
        with st.form("relay_input_form"):
            st.markdown("### 📥 Telemetry Data")
            asset_id = st.text_input("Asset ID / Station Name", value="SUB-RELAY-001")
            
            noise_index = st.number_input(
                "Pre-Fault Noise Index", value=0.05, step=0.01,
                help="Signal fluctuations before the event. High noise (>1.5) indicates analog card issues."
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
            
            run_btn = st.form_submit_button("🔍 Run AI Analysis", use_container_width=True, type="primary")

    with col_dash:
        if run_btn:
            if not os.path.exists('Final_Relay_AI_Model.pkl'):
                st.error("🚨 CRITICAL: Model file 'Final_Relay_AI_Model.pkl' is missing from the directory.")
            else:
                try:
                    model = joblib.load('Final_Relay_AI_Model.pkl')
                    
                    # Safe Dataframe Construction
                    feat_data = dict()
                    noise_list = list()
                    noise_list.append(noise_index)
                    feat_data.update({"Pre_Fault_Noise_Index": noise_list})
                    
                    fault_list = list()
                    fault_list.append(fault_current)
                    feat_data.update({"Max_Fault_Current_A": fault_list})
                    
                    age_list = list()
                    age_list.append(age_months)
                    feat_data.update({"Relay_Age_Months": age_list})
                    
                    alarms_list = list()
                    alarms_list.append(alarms_count)
                    feat_data.update({"Internal_Alarms_Count": alarms_list})
                    
                    features = pd.DataFrame(feat_data)
                    
                    prediction_raw = np.asarray(model.predict(features)).ravel()
                    prediction_idx = int(prediction_raw)
                    
                    probs_array = np.asarray(model.predict_proba(features)).ravel()
                    risk_score = float((1.0 - probs_array) * 100.0)
                    
                    classes = list()
                    classes.append('Safe Condition')
                    classes.append('Analog Card Issue')
                    classes.append('Mechanical Wear')
                    classes.append('CPU/Power Error')
                    mode = classes[prediction_idx]
                    
                    if risk_score < 30:
                        status, color, action = "🟢 SAFE", "#2ecc71", "Normal operation. Maintain TBM schedule."
                    elif risk_score < 70:
                        status, color, action = "🟡 WARNING", "#f1c40f", f"Schedule CBM inspection focusing on {mode}."
                    else:
                        status, color, action = "🔴 DANGER", "#e74c3c", f"Urgent replacement required due to {mode}."

                    st.markdown(f"<div style='border-radius:10px; background:{color}20; border:2px solid {color}; padding:15px; text-align:center;'><h2 style='color:{color}; margin:0;'>{status}: {mode}</h2></div>", unsafe_allow_html=True)
                    
                    # Safe Log Construction
                    log_data = dict()
                    log_data.update({'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                    log_data.update({'Asset ID': asset_id})
                    log_data.update({'Noise Index': noise_index})
                    log_data.update({'Max Current (A)': fault_current})
                    log_data.update({'Risk (%)': round(risk_score, 1)})
                    log_data.update({'Status': status})
                    log_data.update({'Action': action})
                    
                    log_wrap = list()
                    log_wrap.append(log_data)
                    new_log = pd.DataFrame(log_wrap)
                    
                    st.session_state['fmea_log'] = pd.concat([new_log, st.session_state['fmea_log']], ignore_index=True).head(100)
                    st.session_state['latest_eval'] = {'id': asset_id, 'risk': risk_score, 'status': status, 'color': color, 'mode': mode, 'action': action, 'probs': probs_array, 'classes': classes}

                    col_sizes4 = list()
                    col_sizes4.append(1)
                    col_sizes4.append(1)
                    v1, v2 = st.columns(col_sizes4)
                    
                    with v1:
                        st.plotly_chart(create_gauge(risk_score, "Failure Risk Score", color), use_container_width=True)
                    with v2:
                        st.markdown("#### 📊 Failure Mode Probabilities")
                        colors_list = list()
                        colors_list.append('#2ecc71')
                        colors_list.append('#f1c40f')
                        colors_list.append('#e67e22')
                        colors_list.append('#e74c3c')
                        
                        fig_p = px.bar(x=probs_array*100, y=classes, orientation='h', color=classes, color_discrete_sequence=colors_list)
                        fig_p.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Probability (%)", yaxis_title="")
                        st.plotly_chart(fig_p, use_container_width=True)

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
        else:
            st.info("💡 Please input telemetry data and click 'Run AI Analysis' to start.")

with tab2:
    st.markdown("### 📑 RPN & FMEA History Registry")
    if not st.session_state['fmea_log'].empty:
        st.dataframe(st.session_state['fmea_log'].style.apply(highlight_fmea, axis=1), use_container_width=True, hide_index=True)
    else:
        st.info("No records available. Run an assessment to populate the database.")

with tab3:
    if st.session_state['latest_eval']:
        ev = st.session_state['latest_eval']
        st.markdown(f"### 📄 AI Executive Report - {ev['id']}")
        st.info(f"**Health Index:** {100-ev['risk']:.1f}% | **Failure Mode:** {ev['mode']}")
        st.success(f"**Corrective Action:** {ev['action']}")
        report = f"RELAYGUARD AI DIAGNOSTIC REPORT\nAsset: {ev['id']}\nRisk Score: {ev['risk']:.1f}%\nStatus: {ev['status']}\nPrimary Issue: {ev['mode']}\nRecommendation: {ev['action']}"
        st.download_button("📥 Download Full Report (TXT)", data=report, file_name=f"RelayGuard_{ev['id']}.txt", use_container_width=True)
    else:
        st.info("💡 Run an evaluation to generate a report.")