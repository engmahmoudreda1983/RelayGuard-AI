# ==============================================================================
# © 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All rights reserved.
# Proprietary and Confidential. Hybrid AI Expert System Edition.
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
    .report-reasoning {
        background-color: #F4F6F7; padding: 20px; border-radius: 10px; border-left: 5px solid #2980B9; font-size: 16px; margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Login System (STRICTLY FROM SECRETS)
# ==========================================
if not hasattr(st.session_state, 'logged_in'):
    st.session_state.logged_in = False

if not hasattr(st.session_state, 'fmea_log'):
    cols_fmea = list()
    cols_fmea.append('Timestamp')
    cols_fmea.append('Asset ID')
    cols_fmea.append('Noise Index')
    cols_fmea.append('Max Current (A)')
    cols_fmea.append('Risk (%)')
    cols_fmea.append('RPN')
    cols_fmea.append('Status')
    cols_fmea.append('Action')
    st.session_state.fmea_log = pd.DataFrame(columns=cols_fmea)

if not hasattr(st.session_state, 'latest_eval'):
    st.session_state.latest_eval = None

# STRICT SECURITY
try:
    VALID_USER = st.secrets.get("admin_user")
    VALID_PASS = st.secrets.get("admin_pass")
except Exception:
    st.error("🚨 SECURITY ERROR: admin_user or admin_pass NOT found in Secrets Dashboard. Access Denied.")
    st.stop()

if not getattr(st.session_state, 'logged_in'):
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(tuple((1, 1.5, 1)))
    with col2:
        st.markdown("<div class='main-header'><h2 style='text-align: center; margin:0;'>🛡️ RelayGuard AI Login</h2><p style='text-align: center; margin:0; color:#aaa;'>Hybrid AI Expert System</p></div>", unsafe_allow_html=True)
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
            axis=dict(range=tuple((0, 100)), tickwidth=1),
            bar=dict(color=color),
            bgcolor="white",
            steps=tuple((
                dict(range=tuple((0, 30)), color='rgba(46, 204, 113, 0.2)'),
                dict(range=tuple((30, 70)), color='rgba(241, 196, 15, 0.2)'),
                dict(range=tuple((70, 100)), color='rgba(231, 76, 60, 0.2)')
            )),
            threshold=dict(line=dict(color="black", width=4), thickness=0.75, value=value)
        )
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def create_radar(probs_list, classes_tup, color):
    fig = go.Figure(data=go.Scatterpolar(
        r=tuple((p * 100 for p in probs_list)),
        theta=classes_tup,
        fill='toself',
        line_color=color,
        opacity=0.6
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=tuple((0, 100)))),
        showlegend=False,
        height=350,
        margin=dict(l=30, r=30, t=30, b=30)
    )
    return fig

# ==========================================
# 4. Main Interface Logic
# ==========================================
col_title, col_logout = st.columns(tuple((10, 2)))
with col_title:
    st.markdown("<div class='main-header' style='padding: 15px;'><h2 style='margin:0;'>🛡️ RelayGuard AI - Health Dashboard</h2></div>", unsafe_allow_html=True)
with col_logout:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Logout 🚪", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

tab1, tab2, tab3 = st.tabs(tuple(("🛡️ Diagnostic Dashboard", "📑 FMEA & Risk Logs", "📄 Executive Report")))

with tab1:
    col_in, col_dash = st.columns(tuple((1, 2.5)))
    
    with col_in:
        with st.form("relay_input_form"):
            st.markdown("### 📥 Telemetry Data")
            asset_id = st.text_input("Asset ID / Station Name", value="SUB-RELAY-001")
            
            noise_index = st.number_input(
                "Pre-Fault Noise Index (Signal Fluctuation)", value=0.05, step=0.01,
                help="❓ Measures signal noise before a fault. Values > 1.5 strongly indicate hardware deterioration in the Analog Card."
            )
            fault_current = st.number_input(
                "Max Fault Current (A)", value=1500, step=100,
                help="❓ The peak current recorded. Values > 5000A start adding physical stress penalties to the final risk score."
            )
            age_months = st.number_input(
                "Relay Age (Months)", value=192, step=12,
                help="❓ Relays older than 120 months (10 Years) receive an automatic +5% risk penalty for every additional year."
            )
            alarms_count = st.number_input(
                "Internal Alarms Count", value=0, step=1,
                help="❓ Internal self-test failures. Each alarm adds an automatic +1% risk penalty to the assessment."
            )
            
            run_btn = st.form_submit_button("🔍 Run Hybrid AI Analysis", use_container_width=True, type="primary")

    with col_dash:
        if run_btn:
            if not os.path.exists('Final_Relay_AI_Model.pkl'):
                st.error("🚨 CRITICAL: Model file 'Final_Relay_AI_Model.pkl' is missing.")
            else:
                try:
                    model = joblib.load('Final_Relay_AI_Model.pkl')
                    
                    features = pd.DataFrame(
                        tuple((tuple((noise_index, fault_current, age_months, alarms_count)),)), 
                        columns=tuple(('Pre_Fault_Noise_Index', 'Max_Fault_Current_A', 'Relay_Age_Months', 'Internal_Alarms_Count'))
                    )
                    
                    # 1. AI Base Prediction
                    pred_raw = model.predict(features)
                    pred_list = np.ravel(pred_raw).tolist()
                    prediction_idx = int(pred_list.pop(0))
                    
                    probs_array = np.ravel(model.predict_proba(features))
                    base_probs_list = probs_array.tolist()
                    
                    classes_dict = dict()
                    classes_dict.update({0: 'Safe Condition'})
                    classes_dict.update({1: 'Analog Card Issue'})
                    classes_dict.update({2: 'Mechanical Wear'})
                    classes_dict.update({3: 'CPU/Power Error'})
                    
                    actual_classes = getattr(model, 'classes_', tuple(range(len(base_probs_list))))
                    actual_classes_list = list(actual_classes)
                    class_probs_map = dict(zip(actual_classes_list, base_probs_list))
                    
                    ai_safe_prob = float(class_probs_map.get(0, 0.0))
                    ai_base_risk = float((1.0 - ai_safe_prob) * 100.0)
                    
                    # 2. PHYSICAL ENGINEERING BOUNDS (HEURISTICS)
                    age_penalty = max(0.0, float(age_months) - 120.0) / 12.0 * 5.0
                    alarm_penalty = float(alarms_count) * 1.0
                    current_penalty = max(0.0, float(fault_current) - 5000.0) / 1000.0 * 2.0
                    
                    total_penalty = age_penalty + alarm_penalty + current_penalty
                    final_risk = min(100.0, ai_base_risk + total_penalty)
                    
                    # 3. RECALIBRATE PROBABILITIES FOR CONSISTENCY
                    final_probs_list = list(base_probs_list) # Copy original
                    if total_penalty > 0 and 0 in actual_classes_list:
                        safe_idx = actual_classes_list.index(0)
                        # Reduce safe probability based on the total penalty
                        penalty_as_prob = total_penalty / 100.0
                        new_safe_prob = max(0.0, base_probs_list[safe_idx] - penalty_as_prob)
                        final_probs_list[safe_idx] = new_safe_prob
                        
                        # Distribute the extracted penalty to the highest predicted fault or a heuristic fault
                        diff = base_probs_list[safe_idx] - new_safe_prob
                        if diff > 0:
                            # If age is the dominant penalty, assign the risk there (or mechanical wear)
                            target_idx = 2 # Default to Mechanical Wear for physical stress
                            if age_penalty > current_penalty and age_penalty > alarm_penalty:
                                # We'll just push it to mechanical wear as a proxy for aging
                                target_idx = actual_classes_list.index(2) if 2 in actual_classes_list else 1
                            elif alarm_penalty > current_penalty:
                                target_idx = actual_classes_list.index(3) if 3 in actual_classes_list else 1
                            
                            final_probs_list[target_idx] += diff
                    
                    # Normalize to ensure sum is 1.0 just in case
                    prob_sum = sum(final_probs_list)
                    if prob_sum > 0:
                        final_probs_list = [p/prob_sum for p in final_probs_list]
                        
                    ai_confidence = float(max(final_probs_list)) * 100.0
                    
                    # Determine Mode based on the newly calibrated probabilities
                    highest_prob_idx = final_probs_list.index(max(final_probs_list))
                    best_class_val = actual_classes_list[highest_prob_idx]
                    mode = classes_dict.get(best_class_val, "Unknown Error")
                    
                    # Fine-tune the text if penalties heavily overrode the AI
                    if total_penalty > ai_base_risk and mode == 'Safe Condition' and final_risk >= 30:
                        if age_penalty > current_penalty:
                            mode = "Age Degradation (Heuristic)"
                        else:
                            mode = "High Stress Limits Exceeded"
                    
                    colors_dict = dict()
                    colors_dict.update({0: '#2ecc71'})
                    colors_dict.update({1: '#f1c40f'})
                    colors_dict.update({2: '#e67e22'})
                    colors_dict.update({3: '#e74c3c'})
                    
                    plot_classes = list()
                    plot_colors = list()
                    for c in actual_classes_list:
                        class_name = classes_dict.get(int(c), "Unknown Error")
                        # If we used the custom heuristic name, ensure it maps to the plot
                        if class_name == 'Mechanical Wear' and 'Age' in mode:
                            plot_classes.append("Age/Stress Degradation")
                        else:
                            plot_classes.append(class_name)
                        plot_colors.append(colors_dict.get(int(c), "#333333"))
                    
                    # RPN & Status Logic
                    rpn_score = int(final_risk * 10)
                    
                    if final_risk < 30:
                        status, color, action = "🟢 SAFE", "#2ecc71", "Normal operation. Maintain TBM schedule."
                        rpn_level = "Low"
                    elif final_risk < 70:
                        status, color, action = "🟡 WARNING", "#f1c40f", f"Schedule CBM inspection focusing on {mode}."
                        rpn_level = "Medium"
                    else:
                        status, color, action = "🔴 DANGER", "#e74c3c", f"Urgent replacement required due to {mode}."
                        rpn_level = "Critical"

                    # UI Display
                    m1, m2, m3, m4 = st.columns(tuple((1, 1, 1, 1)))
                    m1.metric("🤖 Hybrid Confidence", f"{ai_confidence:.1f}%")
                    m2.metric("⚠️ Failure Risk", f"{final_risk:.1f}%")
                    status_clean = status.replace("🟢", "").replace("🟡", "").replace("🔴", "").strip()
                    m3.metric("📈 Status", status_clean)
                    mode_words = mode.split()
                    m4.metric("⚙️ Primary Issue", mode_words.pop(0) if mode_words else "Unknown")

                    st.markdown(f"<div style='border-radius:10px; background:{color}20; border:2px solid {color}; padding:15px; text-align:center; margin-top: 10px; margin-bottom: 10px;'><h2 style='color:{color}; margin:0;'>{status}: {mode}</h2></div>", unsafe_allow_html=True)
                    
                    # Log
                    new_data = dict()
                    new_data.update({'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                    new_data.update({'Asset ID': asset_id})
                    new_data.update({'Noise Index': noise_index})
                    new_data.update({'Max Current (A)': fault_current})
                    new_data.update({'Risk (%)': round(final_risk, 1)})
                    new_data.update({'RPN': rpn_score})
                    new_data.update({'Status': status})
                    new_data.update({'Action': action})
                    
                    new_log = pd.DataFrame(tuple((new_data,)))
                    st.session_state.fmea_log = pd.concat((new_log, st.session_state.fmea_log), ignore_index=True).head(100)
                    
                    # CLEAN AI REASONING FOR REPORT
                    ai_reasons_clean = ""
                    if final_risk < 30:
                        ai_reasons_clean = "Analysis indicates healthy telemetry. All parameters (Noise, Current, Age, Alarms) are within safe operational thresholds. No immediate action required."
                    else:
                        issues = list()
                        if age_penalty > 0: issues.append(f"significant operating age ({age_months} months)")
                        if alarm_penalty > 0: issues.append(f"internal hardware alarms ({alarms_count} events)")
                        if current_penalty > 0: issues.append(f"high fault current exposure ({fault_current}A)")
                        if ai_base_risk > 30 and noise_index > 1.0: issues.append(f"elevated signal noise ({noise_index})")
                        
                        if issues:
                            ai_reasons_clean = f"The Hybrid Expert System detected an elevated failure risk ({final_risk:.1f}%) primarily driven by " + ", and ".join(issues) + ". This combination indicates a progressing degradation in the relay's reliability."
                        else:
                            ai_reasons_clean = f"The system detected an anomaly resulting in a {final_risk:.1f}% risk score, requiring scheduled maintenance."
                    
                    eval_pack = dict()
                    eval_pack.update({'id': asset_id})
                    eval_pack.update({'risk': final_risk})
                    eval_pack.update({'status': status})
                    eval_pack.update({'color': color})
                    eval_pack.update({'mode': mode})
                    eval_pack.update({'action': action})
                    eval_pack.update({'probs': final_probs_list})
                    eval_pack.update({'classes': tuple(plot_classes)})
                    eval_pack.update({'reasoning': ai_reasons_clean})
                    eval_pack.update({'rpn_score': rpn_score})
                    eval_pack.update({'rpn_level': rpn_level})
                    
                    st.session_state.latest_eval = eval_pack

                    # Top Charts
                    v1, v2 = st.columns(tuple((1, 1)))
                    with v1:
                        st.plotly_chart(create_gauge(final_risk, "Hybrid Failure Risk", color), use_container_width=True)
                    with v2:
                        st.markdown("#### 📊 Hybrid Failure Probabilities")
                        fig_p = px.bar(x=final_probs_list, y=plot_classes, orientation='h', color=plot_classes, color_discrete_sequence=plot_colors)
                        fig_p.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Probability", yaxis_title="")
                        st.plotly_chart(fig_p, use_container_width=True)

                    # --- COMPONENT STRESS CHART ---
                    st.markdown("---")
                    st.markdown("#### 📉 Component Stress Analysis (Relative Impact on Health)")
                    
                    noise_stress = min(150.0, (float(noise_index) / 1.5) * 100.0)
                    curr_stress = min(150.0, (float(fault_current) / 8000.0) * 100.0)
                    age_stress = min(150.0, (float(age_months) / 240.0) * 100.0)
                    alarm_stress = min(150.0, (float(alarms_count) / 10.0) * 100.0)
                    
                    stress_names = tuple(("Noise Fluctuation", "Current Overload", "Age Wear", "Internal Alarms"))
                    stress_vals = tuple((noise_stress, curr_stress, age_stress, alarm_stress))
                    stress_colors = tuple(('#9b59b6', '#e74c3c', '#34495e', '#f39c12'))
                    
                    fig_stress = px.bar(
                        x=stress_vals, 
                        y=stress_names, 
                        orientation='h', 
                        color=stress_names,
                        color_discrete_sequence=stress_colors
                    )
                    fig_stress.add_vline(x=100, line_dash="dash", line_color="red", annotation_text="Critical Limit")
                    fig_stress.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Stress Factor (%)", yaxis_title="")
                    st.plotly_chart(fig_stress, use_container_width=True)

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
        else:
            st.info("💡 Please input telemetry data and click 'Run Hybrid AI Analysis' to start.")

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
        rep_color = eval_data.get('color')
        
        st.markdown(f"### 📄 Comprehensive Hybrid AI Executive Report")
        st.markdown(f"**Asset Designation:** `{eval_data.get('id')}` | **Assessment Time:** `{datetime.now().strftime('%Y-%m-%d %H:%M')}`")
        st.markdown("---")
        
        rep_col1, rep_col2 = st.columns(tuple((1, 1)))
        
        with rep_col1:
            st.markdown("#### 🎯 Diagnostic Summary")
            st.info(f"**Overall Health Index:** {100-eval_data.get('risk'):.1f}%")
            
            mode_text = f"**Predicted Failure Mode:** {eval_data.get('mode')}"
            action_text = f"**{eval_data.get('action')}**"
            
            risk_val = eval_data.get('risk')
            if risk_val < 30:
                st.success(mode_text)
                st.success(f"✅ {action_text}")
            elif risk_val < 70:
                st.warning(mode_text)
                st.warning(f"⚠️ {action_text}")
            else:
                st.error(mode_text)
                st.error(f"🚨 {action_text}")
            
            st.markdown(f"**FMEA RPN (Risk Priority Number):** `{eval_data.get('rpn_score')}/1000` ({eval_data.get('rpn_level')} Severity)")
            
            # CLEAN REASONING TEXT
            st.markdown("#### 🧠 Expert System Reasoning")
            st.markdown(f"<div class='report-reasoning'>{eval_data.get('reasoning')}</div>", unsafe_allow_html=True)

        with rep_col2:
            st.markdown("#### 📡 Multidimensional Hybrid Radar")
            radar_fig = create_radar(eval_data.get('probs'), eval_data.get('classes'), rep_color)
            st.plotly_chart(radar_fig, use_container_width=True)
            
            st.markdown("#### 📈 Predictive Risk Trajectory (Simulated)")
            time_labels = tuple(('M-3', 'M-2', 'M-1', 'Current', 'M+1 (Proj)', 'M+2 (Proj)'))
            
            v1 = max(0.0, risk_val - 12.0)
            v2 = max(0.0, risk_val - 5.0)
            v3 = max(0.0, risk_val - 1.5)
            v4 = risk_val
            v5 = min(100.0, risk_val + 8.0)
            v6 = min(100.0, risk_val + 20.0)
            
            trend_vals = tuple((v1, v2, v3, v4, v5, v6))
            
            trend_fig = go.Figure()
            trend_scatter = go.Scatter(
                x=time_labels, 
                y=trend_vals, 
                mode='lines+markers',
                line=dict(color=rep_color, width=4),
                marker=dict(size=10)
            )
            trend_fig.add_trace(trend_scatter)
            trend_fig.update_layout(
                height=250, 
                margin=dict(l=0, r=0, t=30, b=0), 
                yaxis_title="Risk (%)", 
                xaxis_title="Timeline"
            )
            st.plotly_chart(trend_fig, use_container_width=True)
            
        st.markdown("---")
        report_txt = f"RELAYGUARD HYBRID AI DIAGNOSTIC REPORT\n"
        report_txt += f"=============================================\n"
        report_txt += f"Asset: {eval_data.get('id')}\n"
        report_txt += f"Risk Score: {eval_data.get('risk'):.1f}%\n"
        report_txt += f"RPN Level: {eval_data.get('rpn_score')} ({eval_data.get('rpn_level')})\n"
        report_txt += f"Status: {eval_data.get('status')}\n"
        report_txt += f"Primary Issue: {eval_data.get('mode')}\n"
        report_txt += f"---------------------------------------------\n"
        report_txt += f"System Reasoning: {eval_data.get('reasoning')}\n"
        report_txt += f"---------------------------------------------\n"
        report_txt += f"Mandatory Action: {eval_data.get('action')}\n"
        
        st.download_button("📥 Download Official Report (TXT)", data=report_txt, file_name=f"RelayGuard_{eval_data.get('id')}.txt", use_container_width=True)
    else:
        st.info("💡 Run an evaluation in the Diagnostic Dashboard to generate the Executive Report.")