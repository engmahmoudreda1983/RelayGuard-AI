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
if not hasattr(st.session_state, 'logged_in'):
    st.session_state.logged_in = False

if not hasattr(st.session_state, 'fmea_log'):
    st.session_state.fmea_log = pd.DataFrame(columns=('Timestamp', 'Asset ID', 'Noise Index', 'Max Current (A)', 'Risk (%)', 'Status', 'Action'))

if not hasattr(st.session_state, 'latest_eval'):
    st.session_state.latest_eval = None

# STRICT SECURITY: Access credentials ONLY from Streamlit Secrets Dashboard
try:
    VALID_USER = st.secrets.get("admin_user")
    VALID_PASS = st.secrets.get("admin_pass")
except Exception:
    st.error("🚨 SECURITY ERROR: admin_user or admin_pass NOT found in Secrets Dashboard. Access Denied.")
    st.stop()

if not getattr(st.session_state, 'logged_in'):
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns((1, 1.5, 1))
    with col2:
        st.markdown("<div class='main-header'><h2 style='text-align: center; margin:0;'>🛡️ RelayGuard AI Login</h2><p style='text-align: center; margin:0; color:#aaa;'>Enterprise Protection Analytics Portal</p></div>", unsafe_allow_html=True)
        with st.form("login_form"):
            st.warning("🔒 Restricted Access. Authorized Personnel Only.")
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Secure Login", use_container_width=True):
                if user == VALID_USER and pwd == VALID_PASS:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("❌ Invalid Username or Password")
    st.stop()

# ==========================================
# 3. Helper Functions & Visualization
# ==========================================
def highlight_fmea(row):
    status = row.get('Status', '')
    if 'SAFE' in status: return tuple(('background-color: #E8F5E9; color: #00A65A; font-weight: bold',)) * len(row)
    if 'WARNING' in status: return tuple(('background-color: #FEF9E7; color: #D35400; font-weight: bold',)) * len(row)
    if 'DANGER' in status: return tuple(('background-color: #FDEDEC; color: #C0392B; font-weight: bold',)) * len(row)
    return tuple(('',)) * len(row)

def create_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number=dict(suffix="%", font=dict(size=30, color=color)),
        title=dict(text=title, font=dict(size=18, color='#333')),
        gauge=dict(
            axis=dict(range=(0, 100), tickwidth=1),
            bar=dict(color=color),
            bgcolor="white",
            steps=tuple((
                dict(range=(0, 30), color='rgba(46, 204, 113, 0.2)'),
                dict(range=(30, 70), color='rgba(241, 196, 15, 0.2)'),
                dict(range=(70, 100), color='rgba(231, 76, 60, 0.2)')
            )),
            threshold=dict(line=dict(color="black", width=4), thickness=0.75, value=value)
        )
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ==========================================
# 4. Main Interface Logic
# ==========================================
col_title, col_logout = st.columns((10, 2))
with col_title:
    st.markdown("<div class='main-header' style='padding: 15px;'><h2 style='margin:0;'>🛡️ RelayGuard AI - Health Dashboard</h2></div>", unsafe_allow_html=True)
with col_logout:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Logout 🚪", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

tab1, tab2, tab3 = st.tabs(("🛡️ Diagnostic Dashboard", "📑 FMEA & Risk Logs", "📄 Executive Report"))

with tab1:
    col_in, col_dash = st.columns((1, 2.5))
    
    with col_in:
        with st.form("relay_input_form"):
            st.markdown("### 📥 Telemetry Data")
            asset_id = st.text_input("Asset ID / Station Name", value="SUB-RELAY-001")
            
            noise_index = st.number_input("Pre-Fault Noise Index", value=0.05, step=0.01)
            fault_current = st.number_input("Max Fault Current (A)", value=1500, step=100)
            age_months = st.number_input("Relay Age (Months)", value=192, step=12)
            alarms_count = st.number_input("Internal Alarms Count", value=0, step=1)
            
            run_btn = st.form_submit_button("🔍 Run AI Analysis", use_container_width=True, type="primary")

    with col_dash:
        if run_btn:
            if not os.path.exists('Final_Relay_AI_Model.pkl'):
                st.error("🚨 CRITICAL: Model file 'Final_Relay_AI_Model.pkl' is missing from the directory.")
            else:
                try:
                    # 1. Load Model
                    model = joblib.load('Final_Relay_AI_Model.pkl')
                    
                    # 2. Extract Data Safely
                    features = pd.DataFrame(
                        tuple((tuple((noise_index, fault_current, age_months, alarms_count)),)), 
                        columns=('Pre_Fault_Noise_Index', 'Max_Fault_Current_A', 'Relay_Age_Months', 'Internal_Alarms_Count')
                    )
                    
                    # 3. Model Prediction (Safe 0-dimensional fix)
                    pred_raw = model.predict(features)
                    prediction_idx = int(np.ravel(pred_raw))
                    
                    probs_array = np.ravel(model.predict_proba(features))
                    
                    # NEW: AI Confidence
                    ai_confidence = float(np.max(probs_array)) * 100.0
                    
                    classes_dict = dict()
                    classes_dict.update({0: 'Safe Condition'})
                    classes_dict.update({1: 'Analog Card Issue'})
                    classes_dict.update({2: 'Mechanical Wear'})
                    classes_dict.update({3: 'CPU/Power Error'})
                    
                    colors_dict = dict()
                    colors_dict.update({0: '#2ecc71'})
                    colors_dict.update({1: '#f1c40f'})
                    colors_dict.update({2: '#e67e22'})
                    colors_dict.update({3: '#e74c3c'})
                    
                    # 4. FIX: Dynamic classes mapping matching Model output length
                    actual_classes = getattr(model, 'classes_', tuple(range(len(probs_array))))
                    plot_classes = list()
                    plot_colors = list()
                    for c in actual_classes:
                        plot_classes.append(classes_dict.get(int(c), f"Class {c}"))
                        plot_colors.append(colors_dict.get(int(c), "#333333"))
                        
                    # Calculate Risk safely
                    if 0 in actual_classes:
                        safe_idx_in_array = list(actual_classes).index(0)
                        safe_prob = float(probs_array[safe_idx_in_array])
                    else:
                        safe_prob = float(probs_array)
                        
                    risk_score = float((1.0 - safe_prob) * 100.0)
                    mode = classes_dict.get(prediction_idx, "Unknown Error")
                    
                    if risk_score < 30:
                        status, color, action = "🟢 SAFE", "#2ecc71", "Normal operation. Maintain TBM schedule."
                    elif risk_score < 70:
                        status, color, action = "🟡 WARNING", "#f1c40f", f"Schedule CBM inspection focusing on {mode}."
                    else:
                        status, color, action = "🔴 DANGER", "#e74c3c", f"Urgent replacement required due to {mode}."

                    # --- NEW: KEY NUMBERS METRICS ---
                    m1, m2, m3, m4 = st.columns((1, 1, 1, 1))
                    m1.metric("🤖 AI Confidence", f"{ai_confidence:.1f}%")
                    m2.metric("⚠️ Failure Risk", f"{risk_score:.1f}%")
                    m3.metric("📈 Status", status.replace("🟢", "").replace("🟡", "").replace("🔴", "").strip())
                    m4.metric("⚙️ Primary Issue", mode.split())

                    st.markdown(f"<div style='border-radius:10px; background:{color}20; border:2px solid {color}; padding:15px; text-align:center; margin-top: 10px; margin-bottom: 10px;'><h2 style='color:{color}; margin:0;'>{status}: {mode}</h2></div>", unsafe_allow_html=True)
                    
                    # Log History
                    new_data = {'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Asset ID': asset_id, 'Noise Index': noise_index, 'Max Current (A)': fault_current, 'Risk (%)': round(risk_score, 1), 'Status': status, 'Action': action}
                    new_log = pd.DataFrame(tuple((new_data,)))
                    st.session_state.fmea_log = pd.concat((new_log, st.session_state.fmea_log), ignore_index=True).head(100)
                    st.session_state.latest_eval = dict(id=asset_id, risk=risk_score, status=status, color=color, mode=mode, action=action)

                    # Charts
                    v1, v2 = st.columns((1, 1))
                    with v1:
                        st.plotly_chart(create_gauge(risk_score, "Failure Risk Score", color), use_container_width=True)
                    with v2:
                        st.markdown("#### 📊 Failure Mode Probabilities")
                        fig_p = px.bar(x=probs_array*100, y=plot_classes, orientation='h', color=plot_classes, color_discrete_sequence=plot_colors)
                        fig_p.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Probability (%)", yaxis_title="")
                        st.plotly_chart(fig_p, use_container_width=True)

                    # --- NEW: DYNAMIC VARIABLES BAR CHART ---
                    st.markdown("---")
                    st.markdown("#### 📉 Dynamic Input Variables Comparison")
                    var_names_all = ("Noise Index", "Max Current (A)", "Age (Months)", "Alarms Count")
                    var_vals_all = (noise_index, fault_current, age_months, alarms_count)
                    var_colors = ('#9b59b6', '#e74c3c', '#34495e', '#f39c12')
                    
                    fig_vars = px.bar(
                        x=var_vals_all, 
                        y=var_names_all, 
                        orientation='h', 
                        text=var_vals_all,
                        color=var_names_all,
                        color_discrete_sequence=var_colors
                    )
                    # Log scale applied to X-axis because of large difference between Current (1500) and Noise (0.05)
                    fig_vars.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=30, b=0), xaxis_type="log", xaxis_title="Value (Log Scale)")
                    st.plotly_chart(fig_vars, use_container_width=True)

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
        else:
            st.info("💡 Please input telemetry data and click 'Run AI Analysis' to start.")

with tab2:
    st.markdown("### 📑 RPN & FMEA History Registry")
    fmea_df = getattr(st.session_state, 'fmea_log')
    if not fmea_df.empty:
        st.dataframe(fmea_df.style.apply(highlight_fmea, axis=1), use_container_width=True, hide_index=True)
    else:
        st.info("No records available. Run an assessment to populate the database.")

with tab3:
    eval_data = getattr(st.session_state, 'latest_eval')
    if eval_data:
        st.markdown(f"### 📄 AI Executive Report - {eval_data.get('id')}")
        st.info(f"**Health Index:** {100-eval_data.get('risk'):.1f}% | **Failure Mode:** {eval_data.get('mode')}")
        st.success(f"**Corrective Action:** {eval_data.get('action')}")
        report = f"RELAYGUARD AI DIAGNOSTIC REPORT\nAsset: {eval_data.get('id')}\nRisk Score: {eval_data.get('risk'):.1f}%\nStatus: {eval_data.get('status')}\nPrimary Issue: {eval_data.get('mode')}\nRecommendation: {eval_data.get('action')}"
        st.download_button("📥 Download Full Report (TXT)", data=report, file_name=f"RelayGuard_{eval_data.get('id')}.txt", use_container_width=True)
    else:
        st.info("💡 Run an evaluation to generate a report.")