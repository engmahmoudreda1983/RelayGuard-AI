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

warnings.filterwarnings("ignore")

# ==========================================
# 1. إعدادات الصفحة والتصميم الأساسي
# ==========================================
st.set_page_config(page_title="RelayGuard AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

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
# 2. نظام تسجيل الدخول (Login System)
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'fmea_log' not in st.session_state:
    st.session_state['fmea_log'] = pd.DataFrame(columns=['Timestamp', 'Asset ID', 'Noise Index', 'Fault Current (A)', 'Overall Risk (%)', 'Status', 'RPN Level', 'Action Required'])

if 'latest_eval' not in st.session_state:
    st.session_state['latest_eval'] = None

# جلب بيانات الدخول (fallback to admin/DBA2026 if secrets not set)
try:
    APP_USER = st.secrets["admin_user"]
    APP_PASS = st.secrets["admin_pass"]
except Exception:
    APP_USER = "admin"
    APP_PASS = "DBA2026"

if not st.session_state['logged_in']:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<div class='main-header'><h2 style='text-align: center; margin:0;'>🛡️ RelayGuard AI System</h2><p style='text-align: center; margin:0; color:#aaa;'>Protection Analytics & FMEA</p></div>", unsafe_allow_html=True)
        with st.form("login_form"):
            st.warning("🔒 Restricted Access. Authorized Personnel Only.")
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Secure Login", use_container_width=True)
            if submitted:
                if user == APP_USER and pwd == APP_PASS:
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid Credentials")
    st.stop()

# ==========================================
# 3. دوال مساعدة (Helper Functions)
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
        number={'suffix': "%", 'font': {'size': 30, 'color': color, 'family': "Arial Black"}},
        title={'text': title, 'font': {'size': 18, 'color': '#333'}},
        gauge={
            'axis': {'range':, 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
            'steps': [
                {'range':, 'color': 'rgba(46, 204, 113, 0.3)'}, 
                {'range':, 'color': 'rgba(241, 196, 15, 0.3)'}, 
                {'range':, 'color': 'rgba(231, 76, 60, 0.3)'}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': value}
        }))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

# ==========================================
# 4. الواجهة الرئيسية والتطبيقات (Main Dashboard)
# ==========================================
col_title, col_logout = st.columns()
with col_title:
    st.markdown("""
    <div class="main-header" style="padding: 15px;">
        <h2 style="margin:0;">🛡️ RelayGuard AI - Protection Relay Health Monitor</h2>
        <p style="margin:0; font-size: 14px; color:#CCCCCC;">Predictive Maintenance (CBM) based on COMTRADE & Event Logs Analysis</p>
    </div>
    """, unsafe_allow_html=True)
with col_logout:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Logout 🚪", use_container_width=True):
        st.session_state['logged_in'] = False
        st.rerun()

tab1, tab2, tab3 = st.tabs(["🛡️ Live Assessment & Dashboard", "📑 FMEA & Risk Logs", "📄 Executive AI Report"])

# --- TAB 1: Live Assessment ---
with tab1:
    col_input, col_dashboard = st.columns([1, 2.5])
    
    with col_input:
        with st.form("relay_input_form"):
            st.markdown("### 📥 Telemetry Input")
            asset_id = st.text_input("Asset ID / Station", value="SUB-FADHILI-R01")
            
            noise_index = st.number_input("Pre-Fault Noise Index", min_value=0.0, max_value=10.0, value=0.05, step=0.01)
            fault_current = st.number_input("Max Fault Current (A)", min_value=0, max_value=100000, value=1500, step=100)
            age_months = st.number_input("Relay Age (Months)", min_value=0, max_value=600, value=192, step=12)
            alarms_count = st.number_input("Internal Alarms Count", min_value=0, max_value=50, value=0, step=1)
            
            analyze_btn = st.form_submit_button("🔍 Evaluate Relay Risk", use_container_width=True)

    with col_dashboard:
        if analyze_btn:
            try:
                # 1. تحميل موديل الذكاء الاصطناعي
                relay_model = joblib.load('Final_Relay_AI_Model.pkl')
                
                input_data = pd.DataFrame({
                    'Pre_Fault_Noise_Index': [noise_index],
                    'Max_Fault_Current_A': [fault_current],
                    'Relay_Age_Months': [age_months],
                    'Internal_Alarms_Count': [alarms_count]
                })
                
                # 2. الحسابات والتنبؤ
                prediction_raw = relay_model.predict(input_data)
                prediction_idx = int(np.squeeze(prediction_raw))
                probabilities = relay_model.predict_proba(input_data)
                
                safe_prob = probabilities
                risk_score = (1 - safe_prob) * 100
                
                target_names = ['Safe / Normal', 'Analog Card Degradation', 'Mechanical Wear', 'CPU / Power Issue']
                predicted_mode = target_names[prediction_idx]
                
                # تحديد الحالة والألوان بناءً على المخاطر
                if risk_score < 30:
                    status_text, color, rpn_level, action = "🟢 SAFE", "#2ecc71", "Low", "Continue standard TBM."
                elif risk_score < 70:
                    status_text, color, rpn_level, action = "🟡 WARNING", "#f1c40f", "Medium", f"Schedule CBM. Focus on {predicted_mode}."
                else:
                    status_text, color, rpn_level, action = "🔴 DANGER", "#e74c3c", "Critical", f"URGENT: Replace unit due to {predicted_mode}."

                # 3. حفظ السجل (FMEA Logging)
                new_entry = pd.DataFrame([{
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Asset ID': asset_id,
                    'Noise Index': noise_index,
                    'Fault Current (A)': fault_current,
                    'Overall Risk (%)': round(risk_score, 1),
                    'Status': status_text,
                    'RPN Level': rpn_level,
                    'Action Required': action
                }])
                st.session_state['fmea_log'] = pd.concat([new_entry, st.session_state['fmea_log']], ignore_index=True).head(50)
                
                st.session_state['latest_eval'] = {
                    'asset_id': asset_id, 'risk_score': risk_score, 'status': status_text, 
                    'color': color, 'mode': predicted_mode, 'action': action, 'rpn': rpn_level,
                    'probs': probabilities
                }

                # 4. عرض النتائج والرسوم البيانية
                st.markdown(f"<div style='border: 2px solid {color}; background-color: {color}15; padding: 10px; border-radius: 8px; text-align: center;'><h2 style='color: {color}; margin: 0;'>{status_text}: {predicted_mode}</h2></div><br>", unsafe_allow_html=True)
                
                c1, c2 = st.columns([1, 1.5])
                with c1:
                    st.plotly_chart(create_gauge(risk_score, "Overall Failure Risk", color), use_container_width=True)
                
                with c2:
                    st.markdown("#### 📊 Failure Mode Probabilities")
                    prob_df = pd.DataFrame({'Failure Mode': target_names, 'Probability (%)': probabilities * 100})
                    fig_bar = px.bar(prob_df, x='Probability (%)', y='Failure Mode', orientation='h', 
                                     color='Failure Mode', color_discrete_sequence=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'])
                    fig_bar.update_layout(height=280, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
            except FileNotFoundError:
                st.error("⚠️ AI Model missing! Please ensure 'Final_Relay_AI_Model.pkl' is in the same directory.")
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
        else:
            st.info("👈 Enter relay parameters in the sidebar and click Evaluate to start.")

# --- TAB 2: FMEA & Risk Logs ---
with tab2:
    st.markdown("### 📑 Dynamic FMEA & Risk Registry")
    if not st.session_state['fmea_log'].empty:
        st.dataframe(st.session_state['fmea_log'].style.apply(highlight_fmea, axis=1), use_container_width=True, height=500)
    else:
        st.info("No records found. Run an evaluation to populate the FMEA log.")

# --- TAB 3: Executive AI Report ---
with tab3:
    if st.session_state.get('latest_eval') is None:
        st.info("💡 Please generate a risk assessment from the Live Assessment tab first.")
    else:
        e_data = st.session_state['latest_eval']
        st.markdown(f"""
        <div style='background-color: #F8F9FA; padding: 20px; border-left: 5px solid {e_data['color']}; border-radius: 5px;'>
            <h2 style='margin:0; color: #333;'>📄 RelayGuard AI Executive Diagnostic Report</h2>
            <p style='margin:0; color: #666;'><b>Asset ID:</b> {e_data['asset_id']} | <b>Timestamp:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div><br>
        """, unsafe_allow_html=True)
        
        rep_col1, rep_col2 = st.columns(2)
        with rep_col1:
            st.markdown(f"#### 🎯 Assessment Summary")
            st.markdown(f"- **Overall Risk Score:** <span style='color:{e_data['color']}; font-weight:bold; font-size:18px;'>{e_data['risk_score']:.1f}%</span>", unsafe_allow_html=True)
            st.markdown(f"- **System Status:** {e_data['status']}")
            st.markdown(f"- **Primary Failure Mode:** {e_data['mode']}")
            st.markdown(f"- **RPN Severity Level:** {e_data['rpn']}")
            
        with rep_col2:
            st.markdown(f"#### 📝 Mandatory Action Plan")
            st.markdown(f"<div style='padding:15px; border: 2px dashed {e_data['color']}; background-color: {e_data['color']}10; border-radius:5px;'><b>Control Phase Directive:</b><br>{e_data['action']}</div>", unsafe_allow_html=True)

        st.markdown("---")
        # Text Report Generation for Download
        report_md = f"""=========================================
RELAYGUARD AI - EXECUTIVE DIAGNOSTIC REPORT
=========================================
Asset ID: {e_data['asset_id']}
Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. PREDICTIVE HEALTH SUMMARY
-----------------------------------------
Overall Risk Score: {e_data['risk_score']:.1f}%
System Status: {e_data['status'].replace('🟢', '').replace('🟡', '').replace('🔴', '').strip()}
Primary Failure Mode: {e_data['mode']}

2. DIAGNOSTIC PROBABILITIES
-----------------------------------------
Safe / Normal Condition: {e_data['probs']*100:.1f}%
Analog Card Degradation: {e_data['probs']*100:.1f}%
Mechanical Wear (Contacts): {e_data['probs']*100:.1f}%
CPU / Power Supply Issue: {e_data['probs']*100:.1f}%

3. FMEA DIRECTIVE & ACTION PLAN
-----------------------------------------
RPN Level: {e_data['rpn']}
Mandatory Action: {e_data['action']}

=========================================
Generated by RelayGuard AI Engine
"""
        st.download_button(
            label="📥 Download Full Text Report",
            data=report_md,
            file_name=f"RelayGuard_Report_{e_data['asset_id']}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )