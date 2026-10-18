import hmac
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import relayguard_core as rc

st.set_page_config(page_title="RelayGuard AI", page_icon="🛡️", layout="wide")

STRETCH = "stretch"
NAVY, BLUE = "#0b2e59", "#4da6ff"
GREEN, AMBER, RED = "#1a9e5c", "#d99a00", "#d63a3a"
STATUS_COLOR = {'EXCELLENT': GREEN, 'WATCH': AMBER, 'ACTION': RED}
STATUS_LABEL = {'EXCELLENT': 'NORMAL (Low Failure Risk)', 'WATCH': 'WATCH (Elevated Failure Risk)',
                'ACTION': 'ACTION REQUIRED (High Failure Risk)'}
MODE_ORDER = rc.MODE_ORDER
DEMO_DATE = '2025-10-04'
DEMO_START, DEMO_END = '2025-01-01', '2026-08-31'

st.markdown("""
<style>
.pg-header {background:#0b2e59;padding:16px 22px;border-radius:10px;margin-bottom:18px;display:flex;align-items:center;justify-content:space-between;gap:16px;}
.pg-header img {height:64px;background:#fff;border-radius:8px;padding:4px 8px;}
.pg-header h1 {color:#fff;margin:0;font-size:1.9rem;}
.pg-header p {color:#9fd0ff;margin:4px 0 0 0;font-size:1rem;}
.pg-banner {border:2px solid;border-radius:8px;padding:10px 14px;text-align:center;margin:6px 0 12px 0;}
.pg-banner h2 {margin:0;font-size:1.5rem;}
.pg-banner p {margin:2px 0 0 0;font-size:.95rem;}
.pg-note {background:rgba(77,166,255,.10);border-left:4px solid #4da6ff;padding:8px 12px;border-radius:4px;font-size:.92rem;margin-bottom:10px;}
.pg-card {border:1px solid #d5dce6;border-radius:10px;padding:12px 14px;text-align:center;}
.pg-card h4 {margin:0;color:#0b2e59;font-size:1.05rem;}
.pg-card .big {font-size:1.7rem;font-weight:700;margin:4px 0;}
.pg-card .sub {font-size:.85rem;color:#5a6575;}
.pg-foot {text-align:center;color:#888;font-size:12px;padding:10px;}
</style>
""", unsafe_allow_html=True)

FOOTER = """
<div class='pg-foot'><strong>© 2026 PowerGuard AI by Eng. Mahmoud Reda Ibrahim. All Rights Reserved.</strong><br>
<em>Confidential and Proprietary. The predictive algorithms and expert system overrides within this application are protected intellectual property.</em></div>
"""


def banner(title, sub, color):
    st.markdown(f"<div class='pg-banner' style='border-color:{color}'><h2 style='color:{color}'>{title}</h2>"
                f"<p>{sub}</p></div>", unsafe_allow_html=True)


def secrets_ok():
    try:
        return "APP_USERNAME" in st.secrets and "APP_PASSWORD" in st.secrets
    except Exception:
        return False


# ==========================================
# --- 1. Login ---
# ==========================================
for key, default in [('logged_in', False), ('register', []), ('hist', None)]:
    if key not in st.session_state:
        st.session_state[key] = default

if not st.session_state['logged_in']:
    st.markdown("<br><br><h1 style='text-align:center;color:#4da6ff;'>🛡️ RelayGuard AI Login</h1>", unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.warning("🔒 Restricted Access. Authorized Personnel Only.")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Secure Login", width=STRETCH):
                if not secrets_ok():
                    st.error("Login is not configured: add APP_USERNAME and APP_PASSWORD to the app's Secrets.")
                elif hmac.compare_digest(username, st.secrets["APP_USERNAME"]) and \
                        hmac.compare_digest(password, st.secrets["APP_PASSWORD"]):
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid Username or Password")
    st.markdown(FOOTER, unsafe_allow_html=True)
    st.stop()

# ==========================================
# --- 2. Header and data ---
# ==========================================
_, col_r = st.columns([10, 1])
with col_r:
    if st.button("Logout 🚪"):
        st.session_state['logged_in'] = False
        st.rerun()


@st.cache_data
def logo_b64():
    try:
        import base64, io
        from PIL import Image
        im = Image.open('logo.png')
        im.thumbnail((360, 200))
        buf = io.BytesIO()
        im.save(buf, format='PNG')
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


_logo = logo_b64()
_logo_html = f"<img src='data:image/png;base64,{_logo}' alt='PowerGuard AI'>" if _logo else ""
st.markdown(f"""
<div class='pg-header'>
  <div>
    <h1>🛡️ RelayGuard AI: Protection Relay Condition and Risk Monitor</h1>
    <p>OMICRON test results, self-supervision alarms, event logs, breaker timing and relay age, fused into one risk per relay</p>
  </div>
  {_logo_html}
</div>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def demo_full():
    return rc.demo_data()


@st.cache_data(show_spinner="Assessing every relay, day by day…")
def run_series(register, tests, events, start=None, end=None):
    return rc.assess_series(register, tests, events, start, end)


def demo_series():
    d = demo_full()
    return run_series(d['register'], d['tests'], d['events'], DEMO_START, DEMO_END)


@st.cache_data(show_spinner=False)
def demo_snapshot():
    d = demo_full()
    P = rc.prepare(d['register'], d['tests'], d['events'])
    day = pd.Timestamp(DEMO_DATE)
    return {rid: (r, rc.summary(P, rid, day)) for rid, r in P['reg'].items()}


# ==========================================
# --- 3. Charts ---
# ==========================================
def gauge(value, color, title='Failure Risk (%)', height=290):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value), number={'suffix': '%', 'font': {'size': 34}, 'valueformat': '.0f'},
        title={'text': title, 'font': {'size': 15}}, domain={'x': [0, 1], 'y': [0, 0.85]},
        gauge={'axis': {'range': [0, 100], 'tickvals': [0, 35, 70, 100], 'tickfont': {'size': 12}},
               'bar': {'color': color, 'thickness': 0.3},
               'steps': [{'range': [0, 35], 'color': 'rgba(26,158,92,.25)'},
                         {'range': [35, 70], 'color': 'rgba(217,154,0,.25)'},
                         {'range': [70, 100], 'color': 'rgba(214,58,58,.25)'}]}))
    fig.update_layout(height=height, margin=dict(l=35, r=35, t=40, b=10))
    return fig


def band_color(v):
    return GREEN if v <= rc.EXCELLENT_MAX else AMBER if v <= rc.WATCH_MAX else RED


def rpn_color(r):
    return GREEN if r < 40 else AMBER if r < 100 else RED


def mode_plot(risks, title='Risk by failure mode (risk drivers)', height=300):
    names = sorted(risks, key=lambda k: risks[k])
    vals = [risks[n] for n in names]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation='h', marker_color=[band_color(v) for v in vals],
                           text=[f"{v:.0f}%" for v in vals], textposition='outside', cliponaxis=False,
                           hovertemplate='%{y}: %{x:.1f}%<extra></extra>'))
    for x in (35, 70):
        fig.add_vline(x=x, line_dash='dash', line_color='#888')
    fig.update_layout(title=title, height=height, xaxis=dict(range=[0, 115], title='Risk (%)'), margin=dict(l=10, r=20, t=45, b=40))
    return fig


def rpn_plot(a, height=230):
    rpn = int(a['RPN'])
    fig = go.Figure()
    for x0, x1, c, t in [(0, 40, 'rgba(26,158,92,.18)', 'Low'), (40, 100, 'rgba(217,154,0,.18)', 'Medium'),
                         (100, 200, 'rgba(214,58,58,.18)', 'High')]:
        fig.add_shape(type='rect', x0=x0, x1=x1, y0=-0.5, y1=0.5, fillcolor=c, line_width=0, layer='below')
        fig.add_annotation(x=(x0 + x1) / 2, y=0.62, text=t, showarrow=False, font=dict(size=11, color='#555'))
    fig.add_bar(x=[rpn], y=[0], orientation='h', width=0.45, marker_color=rpn_color(rpn),
                text=[f"RPN {rpn} = S {int(a['S'])} × O {int(a['O'])} × D {int(a['D'])}"],
                textposition='inside' if rpn > 110 else 'outside', insidetextanchor='middle', cliponaxis=False,
                hoverinfo='skip', textfont=dict(color='white' if rpn > 110 else '#222'))
    fig.update_layout(title='RPN of the governing mode (max 200)', height=height, showlegend=False,
                      xaxis=dict(range=[0, 200], tickvals=[0, 40, 100, 200], title='RPN'),
                      yaxis=dict(visible=False, range=[-0.6, 0.8]), margin=dict(l=10, r=10, t=45, b=40))
    return fig


def fmea_rpn_plot(fmea, height=280):
    t = sorted(fmea, key=lambda m: m['RPN'])
    fig = go.Figure(go.Bar(x=[m['RPN'] for m in t], y=[m['Failure Mode'] for m in t], orientation='h',
                           marker_color=[rpn_color(m['RPN']) for m in t], text=[str(m['RPN']) for m in t],
                           textposition='outside', cliponaxis=False, hovertemplate='%{y}: RPN %{x}<extra></extra>'))
    for x in (40, 100):
        fig.add_vline(x=x, line_dash='dash', line_color='#888')
    fig.update_layout(title='RPN by failure mode', height=height, xaxis=dict(range=[0, 200], title='RPN'), margin=dict(l=10, r=20, t=45, b=40))
    return fig


def points_plot(pts, height=None):
    """Error of each test point as % of its tolerance; the drift since the previous test as a marker."""
    if not pts:
        return None
    t = pts[::-1]
    names = [rc.point_name(p) for p in t]
    err = [p['Error (% of tol)'] for p in t]
    fig = go.Figure(go.Bar(x=err, y=names, orientation='h', name='Now',
                           marker_color=[GREEN if e <= 50 else AMBER if e <= 100 else RED for e in err],
                           text=[f"{e:.0f}%" for e in err], textposition='outside', cliponaxis=False,
                           hovertemplate='%{y}: %{x:.0f}% of tolerance<extra></extra>'))
    prev = [(p['Error (% of tol)'] - p['Drift (% of tol)']) if p['Drift (% of tol)'] is not None else None for p in t]
    if any(v is not None for v in prev):
        fig.add_scatter(x=prev, y=names, mode='markers', name='Previous test', marker=dict(symbol='line-ns-open', size=16, color='#333', line=dict(width=2)))
    fig.add_vline(x=100, line_color=RED, line_dash='dash', annotation_text='Tolerance', annotation_position='top')
    fig.add_vline(x=50, line_color='#888', line_dash='dot')
    fig.update_layout(title='Test points: error as % of the tolerance', height=height or (130 + 30 * len(t)),
                      xaxis=dict(range=[0, max(150, max(err) * 1.15)], title='% of tolerance'), yaxis=dict(type='category'),
                      legend=dict(orientation='h', y=-0.25), margin=dict(l=10, r=20, t=50, b=40))
    return fig


def cb_plot(times, rated, height=260):
    fig = go.Figure()
    x = list(range(1, len(times) + 1))
    fig.add_scatter(x=x, y=times, mode='lines+markers', line=dict(color=NAVY), name='Opening time')
    fig.add_hline(y=rated, line_dash='dash', line_color=GREEN, annotation_text=f'Rated {rated:.0f} ms', annotation_position='bottom left')
    fig.add_hline(y=rated * 1.1, line_dash='dot', line_color=AMBER, annotation_text='+10%', annotation_position='top left')
    fig.add_hline(y=rated * 1.5, line_dash='dash', line_color=RED, annotation_text='+50%', annotation_position='top left')
    fig.update_layout(title='Breaker opening time per trip (oldest to newest)', height=height, xaxis=dict(title='Trip', dtick=1),
                      yaxis=dict(title='ms', range=[0, max(rated * 1.7, max(times) * 1.15) if times else rated * 1.7]),
                      showlegend=False, margin=dict(l=45, r=10, t=45, b=40))
    return fig


def ranking_plot(R, top=20):
    t = R.sort_values('Overall Risk (%)', ascending=False).head(top).iloc[::-1]
    fig = go.Figure(go.Bar(x=t['Overall Risk (%)'], y=t['Relay ID'], orientation='h', marker_color=[STATUS_COLOR[s] for s in t['Status']],
                           text=[f"{r:.0f}%  ·  {g}" for r, g in zip(t['Overall Risk (%)'], t['Governing Mode'])],
                           textposition='outside', cliponaxis=False, hovertemplate='%{y}: %{x:.1f}%<extra></extra>'))
    for x, tx in [(35, 'Watch'), (70, 'Action')]:
        fig.add_vline(x=x, line_dash='dash', line_color='#888', annotation_text=tx, annotation_position='top')
    fig.update_layout(title=f"Relay risk ranking (top {min(top, len(R))})", height=140 + 30 * len(t),
                      xaxis=dict(range=[0, 135], title='Failure Risk (%)'), yaxis=dict(type='category'), margin=dict(l=10, r=30, t=60, b=40))
    return fig


def heat_plot(R):
    t = R.sort_values('Overall Risk (%)', ascending=False)
    z = [[r['Mode Risks'][m] for m in MODE_ORDER] for _, r in t.iterrows()]
    fig = go.Figure(go.Heatmap(z=z, x=MODE_ORDER, y=list(t['Relay ID']), zmin=0, zmax=100,
                               colorscale=[[0, '#e8f5ee'], [0.35, '#bfe3cd'], [0.351, '#fbe8b3'], [0.7, '#f2c14e'], [0.701, '#f2a6a6'], [1, RED]],
                               text=[[f"{v:.0f}" for v in row] for row in z], texttemplate='%{text}',
                               hovertemplate='%{y} · %{x}: %{z:.0f}%<extra></extra>', colorbar=dict(title='%', thickness=12)))
    fig.update_layout(title='Risk by relay and failure mode', height=150 + 30 * len(t), yaxis=dict(type='category', autorange='reversed'),
                      xaxis=dict(tickangle=-20), margin=dict(l=10, r=10, t=50, b=70))
    return fig


def gov_plot(R, height=320):
    above = R[R['Status'] != 'EXCELLENT']
    fig = go.Figure()
    for stt, name in [('WATCH', 'Watch'), ('ACTION', 'Action')]:
        fig.add_bar(x=MODE_ORDER, y=[int(((above['Governing Mode'] == m) & (above['Status'] == stt)).sum()) for m in MODE_ORDER],
                    name=name, marker_color=STATUS_COLOR[stt])
    fig.update_layout(barmode='stack', title='Relays above normal by governing failure mode', height=height,
                      yaxis=dict(title='Relays', dtick=1), xaxis=dict(tickangle=-25), legend=dict(orientation='h', y=-0.5),
                      margin=dict(l=40, r=10, t=50, b=40))
    return fig


def fmea_table(a):
    return pd.DataFrame([{'Failure Mode': m['Failure Mode'], 'Risk (%)': round(m['Risk (%)']), 'S': m['S'], 'O': m['O'], 'D': m['D'],
                          'RPN': m['RPN'], 'RPN Level': m['RPN Level']} for m in a['FMEA']])


def points_table(pts):
    return pd.DataFrame([{'Function': p.get('Function'), 'Test Point': p.get('Test Point'), 'Quantity': p.get('Quantity'), 'Unit': p.get('Unit'),
                          'Nominal': p.get('Nominal'), 'Measured': p.get('Measured'), 'Tolerance': p.get('Tolerance'),
                          'Error (% of tol)': round(p['Error (% of tol)']), 'Drift (% of tol)': None if p['Drift (% of tol)'] is None else round(p['Drift (% of tol)']),
                          'Result': 'Pass' if p['Pass'] else 'FAIL'} for p in pts])


def show_result(a, reg, s, k):
    color = STATUS_COLOR[a['Status']]
    banner(STATUS_LABEL[a['Status']], f"{a['Relay ID']}  ·  {a['Bay']} {a['Role']}  ·  governing mode: {a['Governing Mode']}", color)
    pts = a['Test Points']
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test points outside tolerance", f"{sum(1 for p in pts if not p['Pass'])} of {len(pts)}" if pts else "—")
    m2.metric("Breaker opening time", "—" if a['CB Opening Time (ms)'] is None else f"{a['CB Opening Time (ms)']:.0f} ms")
    m3.metric("Relay age", "—" if a['Age (years)'] is None else f"{a['Age (years)']} years")
    m4.metric("Months since last test", "—" if a['Months Since Test'] is None else f"{a['Months Since Test']:.0f}")
    c1, c2 = st.columns([1, 1.3])
    c1.plotly_chart(gauge(a['Overall Risk (%)'], color), width=STRETCH, key=f"{k}_g")
    c2.plotly_chart(mode_plot(a['Mode Risks']), width=STRETCH, key=f"{k}_m")
    st.markdown("#### Why this result")
    for t in rc.findings(a):
        st.markdown(f"- {t}")
    st.warning(f"**Action:** {a['Action']}")
    with st.expander("📐 Test points and breaker timing", expanded=False):
        if pts:
            st.plotly_chart(points_plot(pts), width=STRETCH, key=f"{k}_pt")
            st.dataframe(points_table(pts), hide_index=True, width=STRETCH)
        times = s.get('cb_times') or []
        rated = float(reg.get('CB Rated Opening Time (ms)') or 0)
        if times and rated > 0:
            st.plotly_chart(cb_plot(times, rated), width=STRETCH, key=f"{k}_cbplot")
    st.markdown("#### 🧮 FMEA")
    f1, f2 = st.columns([1.2, 1])
    f1.dataframe(fmea_table(a), hide_index=True, width=STRETCH)
    f1.caption("RPN = Severity × Occurrence × Detection. S = severity of the failure mode; O = mode risk ÷ 10; D = 2. "
               "Low < 40, Medium 40-99, High ≥ 100.")
    with f2:
        st.plotly_chart(rpn_plot(a), width=STRETCH, key=f"{k}_rpn")
        st.plotly_chart(fmea_rpn_plot(a['FMEA']), width=STRETCH, key=f"{k}_fr")


REC_KEYS = ['Relay ID', 'Bay', 'Protected Object', 'Role', 'Technology', 'Breaker ID', 'Governing Mode', 'Overall Risk (%)', 'Status',
            'S', 'O', 'D', 'RPN', 'RPN Level', 'CB Opening Time (ms)', 'Age (years)', 'Months Since Test', 'Action', 'Mode Risks', 'FMEA']


def to_record(a, source):
    rec = {k: a.get(k) for k in REC_KEYS}
    rec['Points Failed'] = sum(1 for p in a['Test Points'] if not p['Pass'])
    rec['Source'] = source
    rec['Assessed'] = datetime.now().strftime('%Y-%m-%d %H:%M') if source == 'Relay page' else pd.Timestamp(a['Assessed On']).strftime('%Y-%m-%d')
    rec['Findings'] = rc.findings(a)
    return rec


@st.cache_data(show_spinner="Analyzing the demo substation…")
def demo_records():
    return [to_record(rc.assess(r, s, DEMO_DATE), 'Demo substation') for r, s in demo_snapshot().values()]


SRC_PAGES, SRC_DEMO = "Results from the analysis pages", "Demo substation (synthetic)"


def current_records():
    if st.session_state.get('ov_src', SRC_PAGES) == SRC_DEMO:
        return demo_records()
    return list(st.session_state['register'])


# ==========================================
# --- 4. Relay page ---
# ==========================================
OBJECTS = ['Line', 'Transformer', 'Busbar', 'Reactor', 'Feeder', 'Generator']
PLAN_OF = {'Line': '21', 'Transformer': '87T', 'Busbar': '87B', 'Reactor': 'EM', 'Feeder': '51', 'Generator': '87T'}
COUNTS = [('Self-test alarm', 'Self-test alarms (90 days)'), ('Relay failure', 'Relay failure / watchdog (90 days)'),
          ('Power supply alarm', 'Power supply alarms (90 days)'), ('Reboot', 'Restarts (90 days)'),
          ('Comms failure', 'Communication failures (30 days)'), ('Time sync loss', 'Time sync losses (30 days)'),
          ('CT supervision', 'CT supervision alarms (30 days)'), ('VT fuse failure', 'VT fuse failures (30 days)'),
          ('Trip circuit supervision', 'Trip circuit supervision alarms (30 days)'), ('Mal-operation', 'Mal-operations (365 days)'),
          ('Breaker failure', 'Breaker failure operations (365 days)'), ('Trips', 'Trips (365 days)')]
DEF = {'id': 'L5-M1', 'bay': 'Line 5', 'obj': 'Line', 'role': 'Main 1', 'funcs': '21, 67N, 79', 'tech': 'Numerical', 'year': 2015,
       'fw': '7.4', 'lfw': '7.4', 'interval': 24, 'cb': 'Q5-CB1', 'rated': 40.0, 'last_test': date(2024, 9, 1), 'cbt': '41, 42, 40', 'day': date.today()}


def plan_points(plan):
    return pd.DataFrame([{'Function': f, 'Test Point': tp, 'Quantity': q, 'Unit': u, 'Nominal': nom, 'Measured': nom, 'Tolerance': tol,
                          'Previous Measured': None} for f, tp, q, u, nom, tol in rc.TEST_PLANS[plan]])


def init_page(p):
    ss = st.session_state
    for k, v in DEF.items():
        ss.setdefault(f'{p}_{k}', v)
    for k, _ in COUNTS:
        ss.setdefault(f'{p}_c_{k}', 0)
    ss.setdefault(f'{p}_pts', plan_points('21'))
    ss.setdefault(f'{p}_ptsver', 0)


def fill_example(p):
    rid = st.session_state.get(f'{p}_ex')
    if not rid or rid == '—':
        return
    r, s = demo_snapshot()[rid]
    ss = st.session_state
    vals = {'id': rid, 'bay': r['Bay'], 'obj': r['Protected Object'], 'role': r['Role'], 'funcs': r['Functions'], 'tech': r['Technology'],
            'year': int(r['Commission Year']), 'fw': str(r['Firmware'] or ''), 'lfw': str(r['Latest Firmware'] or ''),
            'interval': int(r['Test Interval (months)']), 'cb': str(r['Breaker ID'] or ''), 'rated': float(r['CB Rated Opening Time (ms)'] or 0),
            'last_test': pd.Timestamp(s['last_test']).date() if s['last_test'] is not None else DEF['last_test'],
            'cbt': ', '.join(f"{t:g}" for t in s['cb_times'][-6:]), 'day': pd.Timestamp(DEMO_DATE).date()}
    for k, v in vals.items():
        ss[f'{p}_{k}'] = v
    for k, _ in COUNTS:
        ss[f'{p}_c_{k}'] = int(s.get(k, 0))
    ss[f'{p}_pts'] = pd.DataFrame([{c: q.get(c) for c in ['Function', 'Test Point', 'Quantity', 'Unit', 'Nominal', 'Measured', 'Tolerance', 'Previous Measured']}
                                   for q in s['points']])
    ss[f'{p}_ptsver'] += 1


def collect(p, pts_df):
    ss = st.session_state
    g = lambda k: ss.get(f'{p}_{k}', DEF.get(k))
    reg = {'Relay ID': (g('id') or 'RELAY').strip(), 'Bay': g('bay'), 'Protected Object': g('obj'), 'Role': g('role'), 'Functions': g('funcs'),
           'Technology': g('tech'), 'Commission Year': g('year'), 'Firmware': g('fw'), 'Latest Firmware': g('lfw'),
           'Test Interval (months)': g('interval'), 'Breaker ID': g('cb'), 'CB Rated Opening Time (ms)': g('rated')}
    s = {k: int(ss.get(f'{p}_c_{k}', 0)) for k, _ in COUNTS}
    df = pts_df
    s['points'] = [r for r in df.to_dict('records') if rc.isnum(r.get('Nominal')) and rc.isnum(r.get('Measured')) and rc.isnum(r.get('Tolerance'))]
    s['last_test'] = pd.Timestamp(g('last_test')) if g('last_test') else None
    try:
        s['cb_times'] = [float(x) for x in str(g('cbt')).replace(';', ',').split(',') if x.strip()]
    except ValueError:
        s['cb_times'] = []
    return reg, s


def relay_page():
    p = 'rp'
    init_page(p)
    left, right = st.columns([1, 1.7])
    reg_ids = ['—'] + list(demo_full()['register']['Relay ID'])
    with left:
        st.markdown("#### 🛡️ Relay: latest evidence")
        st.selectbox("Fill with a demo relay (optional)", reg_ids, key=f'{p}_ex', on_change=fill_example, args=(p,),
                     format_func=lambda x: x if x == '—' else f"{x} ({DEMO_DATE})")
        st.date_input("Assessment date", key=f'{p}_day', help="The date the counts and the test age refer to (normally today).")
        st.markdown("**Relay data**")
        c1, c2, c3 = st.columns(3)
        c1.text_input("Relay ID", key=f'{p}_id'); c2.text_input("Bay", key=f'{p}_bay'); c3.selectbox("Protected object", OBJECTS, key=f'{p}_obj')
        c1, c2 = st.columns(2)
        c1.text_input("Role", key=f'{p}_role', help="Main 1, Main 2, Main or Backup."); c2.text_input("Functions", key=f'{p}_funcs', help="ANSI numbers, e.g. 21, 67N, 79 or 87L.")
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Technology", ['Numerical', 'Static', 'Electromechanical'], key=f'{p}_tech')
        c2.number_input("Commission year", 1960, 2030, key=f'{p}_year'); c3.number_input("Test interval (months)", 3, 96, key=f'{p}_interval')
        c1, c2 = st.columns(2)
        c1.text_input("Firmware", key=f'{p}_fw'); c2.text_input("Latest firmware", key=f'{p}_lfw')
        st.markdown("**OMICRON test: latest session**")
        st.date_input("Test date", key=f'{p}_last_test')
        st.caption("One row per test point. 'Previous Measured' (optional) is the value from the previous test, for the drift.")
        pts_df = st.data_editor(st.session_state[f'{p}_pts'], num_rows='dynamic', key=f"{p}_ed_{st.session_state[f'{p}_ptsver']}", width=STRETCH, hide_index=True)
        st.markdown("**Alarms and events (from the event log)**")
        cols = st.columns(2)
        for i, (k, lab) in enumerate(COUNTS):
            cols[i % 2].number_input(lab, 0, 999, key=f'{p}_c_{k}')
        st.markdown("**Breaker**")
        c1, c2 = st.columns(2)
        c1.text_input("Breaker ID (empty = none)", key=f'{p}_cb'); c2.number_input("Rated opening time (ms)", 0.0, 500.0, key=f'{p}_rated')
        st.text_input("Opening times of the last trips (ms, oldest first)", key=f'{p}_cbt', help="Comma-separated, e.g. 41, 43, 47.")
        go_btn = st.button("🔍 Analyze Relay", width=STRETCH, key=f"{p}_go", type='primary')
    with right:
        if go_btn:
            reg, s = collect(p, pts_df)
            a = rc.assess(reg, s, pd.Timestamp(st.session_state[f'{p}_day']))
            st.session_state[f"{p}_last"] = (a, reg, s)
            st.session_state['register'] = [r for r in st.session_state['register'] if r['Relay ID'] != a['Relay ID']]
            st.session_state['register'].append(to_record(a, 'Relay page'))
            st.rerun()
        last = st.session_state.get(f"{p}_last")
        if last is None:
            st.markdown("<div class='pg-note'>Enter the relay data, the latest OMICRON test points, the alarm counts from the event log "
                        "and the breaker opening times on the left (or fill them from a demo relay), then press <b>Analyze Relay</b>. "
                        "The result is added to the Substation Overview and the Combined Report.</div>", unsafe_allow_html=True)
        else:
            show_result(*last, p)
            st.caption("Added to the Substation Overview and the Combined Report.")


# ==========================================
# --- 5. Quick checks ---
# ==========================================
def read_csv(label, key, need):
    f = st.file_uploader(label, type='csv', key=key)
    if f is None:
        return None, None
    try:
        df = pd.read_csv(f)
    except Exception as e:
        return None, f"Cannot read the file: {e}"
    miss = [c for c in need if c not in df.columns]
    return (None, f"Missing columns: {', '.join(miss)}") if miss else (df, None)


NEED = {'register': ['Relay ID', 'Bay', 'Protected Object', 'Role', 'Functions', 'Technology', 'Commission Year', 'Test Interval (months)'],
        'tests': ['Date', 'Relay ID', 'Function', 'Test Point', 'Quantity', 'Nominal', 'Measured', 'Tolerance'],
        'events': ['Timestamp', 'Relay ID', 'Event'], 'findings': ['Relay ID', 'Date', 'Event']}


def test_report_html(rid, d, pts, reg):
    def drift(p):
        return '' if p['Drift (% of tol)'] is None else f"{p['Drift (% of tol)']:+.0f}%"
    rows = ''.join(f"<tr><td>{p.get('Function')}</td><td>{p.get('Test Point')}</td><td>{p.get('Quantity')}</td><td>{p.get('Unit') or ''}</td>"
                   f"<td>{rc.fmt_num(p['Nominal'])}</td><td>{rc.fmt_num(p['Measured'])}</td><td>± {rc.fmt_num(p['Tolerance'])}</td>"
                   f"<td>{p['Error (% of tol)']:.0f}%</td><td>{drift(p)}</td>"
                   f"<td style='color:{'#1a9e5c' if p['Pass'] else '#d63a3a'};font-weight:700'>{'PASS' if p['Pass'] else 'FAIL'}</td></tr>" for p in pts)
    n_f = sum(1 for p in pts if not p['Pass'])
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Relay test report {rid} {d}</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#1f2937}}h1{{color:#0b2e59;margin:0}}table{{border-collapse:collapse;width:100%;margin-top:16px;font-size:13px}}
th{{background:#0b2e59;color:#fff;text-align:left;padding:6px}}td{{border-bottom:1px solid #d5dce6;padding:6px}}.k{{color:#5a6575}}</style></head><body>
<h1>Relay Test Report</h1><p class='k'>RelayGuard AI · generated {datetime.now():%Y-%m-%d %H:%M}</p>
<table><tr><td class='k'>Relay</td><td><b>{rid}</b></td><td class='k'>Bay</td><td>{reg.get('Bay', '')}</td></tr>
<tr><td class='k'>Functions</td><td>{reg.get('Functions', '')}</td><td class='k'>Technology</td><td>{reg.get('Technology', '')}</td></tr>
<tr><td class='k'>Test date</td><td>{d}</td><td class='k'>Result</td><td><b style='color:{'#1a9e5c' if not n_f else '#d63a3a'}'>{'ALL POINTS PASS' if not n_f else f'{n_f} POINT(S) FAIL'}</b></td></tr></table>
<table><tr><th>Function</th><th>Test point</th><th>Quantity</th><th>Unit</th><th>Nominal</th><th>Measured</th><th>Tolerance</th><th>Error (% of tol)</th><th>Drift</th><th>Result</th></tr>{rows}</table>
<p class='k' style='margin-top:18px'>Error = |measured − nominal| ÷ tolerance. Drift = change of the error since the previous test of the same point.</p></body></html>"""


def test_review_page():
    st.markdown("<div class='pg-note'>Review an OMICRON test session: each test point against its tolerance, the drift since the previous "
                "test, and a test report you can download. Quick check only: not added to the overview.</div>", unsafe_allow_html=True)
    src = st.radio("Test results", ["Demo substation files", "Upload OMICRON export (CSV)"], horizontal=True, key='tr_src')
    if src.startswith("Demo"):
        reg_df, t_df = demo_full()['register'], demo_full()['tests']
    else:
        c1, c2 = st.columns(2)
        with c1:
            t_df, e1 = read_csv("OMICRON test points (CSV)", 'tr_t', NEED['tests'])
        with c2:
            reg_df, e2 = read_csv("Relay register (optional)", 'tr_r', ['Relay ID'])
        for e in [e1, e2]:
            if e:
                st.error(e)
        with st.expander("📄 File format"):
            st.markdown("One row per test point: `Date, Relay ID, Function, Test Point, Quantity, Unit, Nominal, Measured, Tolerance, Result` "
                        "(Tolerance in the same unit as Nominal; Result optional). Export the test points from the OMICRON report to CSV.")
            st.download_button("Demo test file", demo_full()['tests'].to_csv(index=False).encode('utf-8-sig'), 'demo_relay_tests.csv', 'text/csv')
        if t_df is None:
            return
        if reg_df is None:
            reg_df = pd.DataFrame({'Relay ID': sorted(t_df['Relay ID'].astype(str).unique())})
    P = rc.prepare(reg_df, t_df, None)
    rids = [r for r in P['reg'] if r in P['tests']]
    if not rids:
        st.error("No test results match the relays.")
        return
    c1, c2 = st.columns(2)
    rid = c1.selectbox("Relay", rids, index=rids.index('L2-M1') if 'L2-M1' in rids else 0, key='tr_rid')
    dates = P['tests'][rid][0]
    sess_list = dates[::-1]
    worst_i = max(range(len(sess_list)), key=lambda i: max([rc.point_error(q) or 0 for q in P['tests'][rid][1][sess_list[i]]] or [0]))
    d = c2.selectbox("Test session", sess_list, index=worst_i, format_func=lambda x: pd.Timestamp(x).strftime('%Y-%m-%d'), key=f'tr_d_{rid}',
                     help="Opens on the session with the largest error.")
    s = rc.summary(P, rid, d)
    reg = P['reg'][rid]
    a = rc.assess({**reg, 'Breaker ID': ''}, {**s, 'Mal-operation': 0}, d)
    pts = a['Test Points']
    n_f = sum(1 for p in pts if not p['Pass'])
    sc = a['Mode Risks']['Measurement and timing']
    banner('ALL TEST POINTS PASS' if not n_f else f'{n_f} TEST POINT(S) OUTSIDE TOLERANCE', f"{rid} · test of {pd.Timestamp(d).date()} · measurement and timing risk {sc:.0f}%",
           band_color(sc))
    st.plotly_chart(points_plot(pts), width=STRETCH, key='tr_pts')
    st.dataframe(points_table(pts), hide_index=True, width=STRETCH)
    for t in a['Reasons']['Measurement and timing']:
        st.markdown(f"- {t}")
    # history of each test point over all sessions
    hist = []
    for dd in dates:
        for p in P['tests'][rid][1][dd]:
            e = rc.point_error(p)
            if e is not None:
                hist.append({'Date': dd, 'Point': rc.point_name(p), 'Error': e * 100})
    if len(dates) > 1:
        H = pd.DataFrame(hist)
        fig = go.Figure()
        for i, (nme, g) in enumerate(H.groupby('Point')):
            fig.add_scatter(x=g['Date'], y=g['Error'], mode='lines+markers', name=nme)
        fig.add_hline(y=100, line_dash='dash', line_color=RED, annotation_text='Tolerance', annotation_position='top left')
        fig.update_layout(title=f'{rid}: error of each test point over the test history', height=380, yaxis=dict(title='% of tolerance', rangemode='tozero'),
                          legend=dict(orientation='h', y=-0.25), margin=dict(l=50, r=20, t=50, b=40))
        st.plotly_chart(fig, width=STRETCH, key='tr_hist')
    st.download_button("📥 Download test report (HTML)", test_report_html(rid, pd.Timestamp(d).date(), pts, reg).encode('utf-8'),
                       f"Relay_Test_Report_{rid}_{pd.Timestamp(d):%Y%m%d}.html", 'text/html')


def events_page():
    st.markdown("<div class='pg-note'>Read the relay event log (SER): self-supervision, communication, secondary-circuit alarms and trips "
                "for one relay, counted in the same windows the engine uses. Quick check only: not added to the overview.</div>", unsafe_allow_html=True)
    src = st.radio("Event log", ["Demo substation files", "Upload event log (CSV)"], horizontal=True, key='ev_src')
    if src.startswith("Demo"):
        reg_df, e_df = demo_full()['register'], demo_full()['events']
        default_day = pd.Timestamp(DEMO_DATE).date()
    else:
        c1, c2 = st.columns(2)
        with c1:
            e_df, e1 = read_csv("Event log (CSV)", 'ev_e', NEED['events'])
        with c2:
            reg_df, e2 = read_csv("Relay register (optional)", 'ev_r', ['Relay ID'])
        for e in [e1, e2]:
            if e:
                st.error(e)
        with st.expander("📄 File format"):
            st.markdown("`Timestamp, Relay ID, Event, Value`. Event names: " + ", ".join(f"`{x}`" for x in rc.EVENT_TYPES) +
                        ". Value = breaker opening time in ms for `Trip`. `Maintenance` = repair completed: older alarms stop counting.")
            st.download_button("Demo event log", demo_full()['events'].to_csv(index=False).encode('utf-8-sig'), 'demo_relay_events.csv', 'text/csv')
        if e_df is None:
            return
        if reg_df is None:
            reg_df = pd.DataFrame({'Relay ID': sorted(e_df['Relay ID'].astype(str).unique())})
        default_day = pd.to_datetime(e_df['Timestamp'], errors='coerce').max().date()
    P = rc.prepare(reg_df, None, e_df)
    rids = list(P['reg'])
    c1, c2 = st.columns(2)
    rid = c1.selectbox("Relay", rids, index=rids.index('T1-87') if 'T1-87' in rids else 0, key='ev_rid')
    day = c2.date_input("Assessment date", default_day, key='ev_day')
    s = rc.summary(P, rid, day)
    reg = P['reg'][rid]
    a = rc.assess({**reg, 'Commission Year': None, 'Firmware': '', 'Latest Firmware': '', 'Technology': 'Numerical'},
                  {**s, 'points': [], 'last_test': pd.Timestamp(day)}, day)
    sub = {m: a['Mode Risks'][m] for m in ['Hardware', 'Communication', 'Secondary circuits', 'Breaker']}
    worst = max(sub, key=sub.get); stt = rc.status_of(sub[worst])
    banner(STATUS_LABEL[stt], f"{rid} · {worst} {sub[worst]:.0f}% · as of {day}", STATUS_COLOR[stt])
    c1, c2 = st.columns([1, 1.3])
    c1.plotly_chart(gauge(sub[worst], STATUS_COLOR[stt], f'{worst} risk (%)'), width=STRETCH, key='ev_g')
    c2.plotly_chart(mode_plot(sub, 'Risk from the event log', 260), width=STRETCH, key='ev_m')
    cnt = pd.DataFrame([{'Event': lab.split(' (')[0], 'Window': lab.split('(')[1].rstrip(')'), 'Count': s.get(k, 0)} for k, lab in COUNTS])
    st.dataframe(cnt, hide_index=True, width=STRETCH)
    if s.get('last_maintenance') is not None:
        st.caption(f"Maintenance recorded on {s['last_maintenance']:%Y-%m-%d}: alarms before that date are not counted.")
    for m in sorted(sub, key=lambda m: -sub[m]):
        st.markdown(f"- **{m} ({sub[m]:.0f}%):** " + ' '.join(a['Reasons'][m]))
    ev = e_df.copy(); ev['_t'] = pd.to_datetime(ev['Timestamp'], errors='coerce')
    ev = ev[(ev['Relay ID'].astype(str) == rid) & (ev['_t'] <= pd.Timestamp(day) + pd.Timedelta(days=1)) & (ev['_t'] > pd.Timestamp(day) - pd.Timedelta(days=365))]
    if len(ev):
        order = [e for e in rc.EVENT_TYPES if e in set(ev['Event'])]
        fig = go.Figure(go.Scatter(x=ev['_t'], y=ev['Event'], mode='markers', marker=dict(size=9, color=NAVY, symbol='line-ns-open', line=dict(width=2)),
                                   hovertemplate='%{x}<br>%{y}<extra></extra>'))
        fig.update_layout(title=f'{rid}: events in the last 12 months', height=140 + 32 * len(order), yaxis=dict(categoryorder='array', categoryarray=order[::-1]),
                          margin=dict(l=10, r=10, t=45, b=40))
        st.plotly_chart(fig, width=STRETCH, key='ev_tl')


def breaker_page():
    st.markdown("<div class='pg-note'>Breaker opening time from the event log (relay trip command to breaker open) against the rated value. "
                "Quick check only: not added to the overview.</div>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.7])
    with left:
        cb = st.text_input("Breaker ID", "Q3-CB1", key='cb_id')
        rated = st.number_input("Rated opening time (ms)", 5.0, 500.0, 40.0, key='cb_rated')
        times = st.text_area("Opening times of the last trips (ms, oldest first)", "41.5, 43.2, 45.0, 47.9, 51.3, 54.0", key='cb_times')
        trips = st.number_input("Trips in the last 12 months", 0, 999, 22, key='cb_trips')
        bf = st.number_input("Breaker failure operations (12 months)", 0, 99, 0, key='cb_bf')
        go_btn = st.button("🔍 Analyze", width=STRETCH, key='cb_go', type='primary')
    with right:
        if go_btn:
            try:
                tl = [float(x) for x in times.replace(';', ',').replace('\n', ',').split(',') if x.strip()]
            except ValueError:
                st.error("Opening times must be numbers separated by commas.")
                return
            reg = {'Relay ID': cb, 'Breaker ID': cb, 'CB Rated Opening Time (ms)': rated, 'Commission Year': None}
            a = rc.assess(reg, {'cb_times': tl[-24:], 'Trips': trips, 'Breaker failure': bf, 'last_test': pd.Timestamp.now()})
            st.session_state['cb_last'] = (a, tl[-24:], rated)
        last = st.session_state.get('cb_last')
        if last is None:
            st.markdown("<div class='pg-note'>Enter the opening times recorded at the last trips and press <b>Analyze</b>.</div>", unsafe_allow_html=True)
            return
        a, tl, rated = last
        v = a['Mode Risks']['Breaker']; stt = rc.status_of(v)
        banner(STATUS_LABEL[stt], f"{a['Breaker ID']} · breaker risk {v:.0f}%", STATUS_COLOR[stt])
        c1, c2 = st.columns([1, 1.4])
        c1.plotly_chart(gauge(v, STATUS_COLOR[stt], 'Breaker risk (%)'), width=STRETCH, key='cb_g')
        if tl:
            c2.plotly_chart(cb_plot(tl, rated), width=STRETCH, key='cb_p')
        for t in a['Reasons']['Breaker']:
            st.markdown(f"- {t}")
        if stt != 'EXCELLENT':
            st.warning(f"**Action:** {rc.ACTIONS['Breaker']}")


# ==========================================
# --- 6. Tabs ---
# ==========================================
tabs = st.tabs(["🏭 Substation Overview", "🛡️ Relay Risk & FMEA", "🧪 OMICRON Test Review", "🚨 Alarms & Events",
                "⚡ Breaker Timing", "📄 Combined Report", "📈 Historical Study"])

with tabs[0]:
    st.radio("Show", [SRC_PAGES, SRC_DEMO], horizontal=True, key='ov_src')
    R0 = current_records()
    if st.session_state['ov_src'] == SRC_PAGES:
        st.markdown("<div class='pg-note'>Every relay analyzed on the <b>Relay Risk & FMEA</b> page is collected here. "
                    "The latest result per relay is kept.</div>", unsafe_allow_html=True)
        if R0 and st.button("🗑 Clear results"):
            st.session_state['register'] = []
            st.rerun()
    else:
        st.info(f"Synthetic demo substation: 4 lines (Main 1 distance, Main 2 line differential), 2 transformers, 2 busbar zones and "
                f"1 reactor, 15 relays; status on {DEMO_DATE}. Not real measurements.")
    if not R0:
        st.markdown("<h3 style='text-align:center;color:#888;margin-top:30px'>💡 No results yet: analyze a relay on the "
                    "Relay Risk & FMEA page, or show the demo substation.</h3>", unsafe_allow_html=True)
    else:
        R = pd.DataFrame(R0).sort_values('Overall Risk (%)', ascending=False).reset_index(drop=True)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Relays", len(R)); k2.metric("🔴 Action required", int((R['Status'] == 'ACTION').sum()))
        k3.metric("🟠 Watch", int((R['Status'] == 'WATCH').sum())); k4.metric("🟢 Normal", int((R['Status'] == 'EXCELLENT').sum()))
        w = R.iloc[0]
        rp = R.sort_values('RPN', ascending=False).iloc[0]
        cbr = R.dropna(subset=['CB Opening Time (ms)'])
        failed = int((R['Points Failed'] > 0).sum())
        overdue = R[R['Mode Risks'].apply(lambda m: m['Test overdue'] > 0)]
        cards = [("Highest risk", f"{w['Overall Risk (%)']:.0f}%", STATUS_COLOR[w['Status']], f"{w['Relay ID']} · {w['Governing Mode']}"),
                 ("Highest RPN", f"{int(rp['RPN'])}", rpn_color(rp['RPN']), f"{rp['Relay ID']} · {rp['RPN Level']}"),
                 ("Relays with failed test points", f"{failed}", RED if failed else GREEN, "latest OMICRON session"),
                 ("Tests overdue", f"{len(overdue)}", AMBER if len(overdue) else GREEN, ', '.join(overdue['Relay ID'].head(4)) or 'none')]
        for col, (t, big, c, sub) in zip(st.columns(4), cards):
            col.markdown(f"<div class='pg-card' style='border-top:5px solid {c}'><h4>{t}</h4><div class='big' style='color:{c}'>{big}</div>"
                         f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)
        st.write("")
        left, right = st.columns([1.1, 1])
        left.plotly_chart(ranking_plot(R), width=STRETCH, key='ov_rank')
        right.plotly_chart(gov_plot(R), width=STRETCH, key='ov_gov')
        st.plotly_chart(heat_plot(R), width=STRETCH, key='ov_heat')
        st.markdown("#### All relays")
        show_cols = ['Relay ID', 'Bay', 'Role', 'Technology', 'Governing Mode', 'Overall Risk (%)', 'Status', 'RPN', 'RPN Level',
                     'Points Failed', 'CB Opening Time (ms)', 'Age (years)', 'Source', 'Assessed', 'Action']
        st.dataframe(R[show_cols].round(1), hide_index=True, width=STRETCH)
        st.download_button("📥 Download relay ranking (CSV)", R[show_cols].round(1).to_csv(index=False).encode('utf-8-sig'),
                           f"RelayGuard_overview_{datetime.now():%Y%m%d}.csv", 'text/csv')

with tabs[1]:
    relay_page()
with tabs[2]:
    test_review_page()
with tabs[3]:
    events_page()
with tabs[4]:
    breaker_page()

with tabs[5]:
    recs = current_records()
    if not recs:
        st.markdown("<h3 style='text-align:center;color:#888;margin-top:40px'>💡 Analyze relays on the Relay Risk & FMEA page "
                    "(or show the demo substation on the Overview page) first.</h3>", unsafe_allow_html=True)
    else:
        R = pd.DataFrame(recs).sort_values('Overall Risk (%)', ascending=False).reset_index(drop=True)
        n_a, n_w = int((R['Status'] == 'ACTION').sum()), int((R['Status'] == 'WATCH').sum())
        st.markdown(f"<div class='pg-note'><b>📄 Combined Protection Condition Report</b><br>{len(R)} relays · source: "
                    f"{st.session_state.get('ov_src', SRC_PAGES).lower()} · generated {datetime.now():%Y-%m-%d %H:%M}</div>", unsafe_allow_html=True)
        top = R.iloc[0]
        if n_a:
            summary_txt = (f"{n_a} relay(s) need action and {n_w} are on watch. Highest risk: {top['Relay ID']} "
                           f"({top['Governing Mode']}, {top['Overall Risk (%)']:.0f}%). {top['Action']}")
        elif n_w:
            summary_txt = f"No relay needs immediate action; {n_w} relay(s) are on watch. Highest: {top['Relay ID']} ({top['Governing Mode']})."
        else:
            summary_txt = "All assessed relays are within normal limits."
        st.markdown(f"**Executive summary:** {summary_txt}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Relays", len(R)); k2.metric("🔴 Action", n_a); k3.metric("🟠 Watch", n_w); k4.metric("🟢 Normal", len(R) - n_a - n_w)
        c1, c2 = st.columns([1.1, 1])
        c1.plotly_chart(ranking_plot(R), width=STRETCH, key='rep_rank')
        c2.plotly_chart(gov_plot(R), width=STRETCH, key='rep_gov')
        cols = ['Relay ID', 'Bay', 'Role', 'Governing Mode', 'Overall Risk (%)', 'Status', 'RPN', 'RPN Level', 'Source', 'Assessed', 'Action']
        st.markdown("#### All relays")
        st.dataframe(R[cols].round(1), hide_index=True, width=STRETCH)
        st.markdown("#### Relays above the normal band")
        above = R[R['Status'] != 'EXCELLENT']
        if above.empty:
            st.success("No relay above the normal band.")
        for i, (_, r) in enumerate(above.iterrows()):
            with st.expander(f"{r['Relay ID']}  ·  {r['Governing Mode']}  ·  {r['Overall Risk (%)']:.0f}% {r['Status']}", expanded=i < 3):
                e1, e2 = st.columns([1, 1.1])
                with e1:
                    for t in r['Findings']:
                        st.markdown(f"- {t}")
                    st.markdown(f"**RPN** = {r['S']} × {r['O']} × {r['D']} = **{r['RPN']}** ({r['RPN Level']})")
                e2.plotly_chart(mode_plot(r['Mode Risks'], height=280), width=STRETCH, key=f"rep_m_{i}")
                st.warning(f"**Action:** {r['Action']}")
        lines = ["=========================================", "RELAYGUARD AI - COMBINED PROTECTION CONDITION REPORT",
                 "=========================================", f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
                 f"Relays: {len(R)}  (action {n_a}, watch {n_w})", "", "EXECUTIVE SUMMARY:", summary_txt, ""]
        for _, r in R.iterrows():
            lines += ["-----------------------------------------",
                      f"{r['Relay ID']} ({r['Bay']}, {r['Role']}) - {r['Status']} - failure risk {r['Overall Risk (%)']:.0f}%",
                      f"Governing mode: {r['Governing Mode']}  |  RPN = {r['S']} x {r['O']} x {r['D']} = {r['RPN']} ({r['RPN Level']})",
                      "Mode risks: " + ", ".join(f"{m} {v:.0f}%" for m, v in r['Mode Risks'].items()),
                      f"Source: {r['Source']} ({r['Assessed']})"] + [f"* {t}" for t in r['Findings']] + [f"Action: {r['Action']}"]
        lines += ["=========================================", "Generated by RelayGuard AI Engine"]
        fm = pd.DataFrame([dict(**{'Relay ID': r['Relay ID']}, **m) for _, r in R.iterrows() for m in r['FMEA']])
        d1, d2, d3 = st.columns(3)
        d1.download_button("📥 Download Full Text Report", "\n".join(lines).encode('utf-8'), f"RelayGuard_Report_{datetime.now():%Y%m%d}.txt", 'text/plain')
        d2.download_button("📥 Download Table (CSV)", R[cols].round(1).to_csv(index=False).encode('utf-8-sig'), f"RelayGuard_Report_{datetime.now():%Y%m%d}.csv", 'text/csv')
        d3.download_button("📥 Download FMEA, all modes (CSV)", fm.to_csv(index=False).encode('utf-8-sig'), f"RelayGuard_FMEA_{datetime.now():%Y%m%d}.csv", 'text/csv')

with tabs[6]:
    st.markdown("<div class='pg-note'>Replays the history day by day. Each day every relay is assessed with only the data that existed on "
                "that day (latest test session, alarm windows, breaker trips), then the alerts are compared with the recorded failures and "
                "findings. The result shows how early the system would have warned.</div>", unsafe_allow_html=True)
    hsrc = st.radio("Data", ["Demo substation (synthetic, Jan 2025 - Aug 2026)", "Upload client files"], horizontal=True, key='h_src')
    files, errs = {}, []
    if hsrc.startswith("Upload"):
        lab = {'register': 'Relay register (required)', 'tests': 'OMICRON test points (required)', 'events': 'Event log / SER (required)',
               'findings': 'Findings: failures and major findings'}
        cols = st.columns(2)
        for i, k in enumerate(NEED):
            with cols[i % 2]:
                files[k], e = read_csv(lab[k], f'h_{k}', NEED[k])
                if e:
                    errs.append(f"{k}: {e}")
        with st.expander("📄 File formats and demo files"):
            st.markdown("""
- **Relay register:** `Relay ID, Bay, Protected Object, Role, Functions, Technology (Numerical/Static/Electromechanical), Commission Year,
  Firmware, Latest Firmware, Test Interval (months), Breaker ID, CB Rated Opening Time (ms)`.
- **OMICRON test points:** `Date, Relay ID, Function, Test Point, Quantity, Unit, Nominal, Measured, Tolerance, Result`.
- **Event log (SER):** `Timestamp, Relay ID, Event, Value` (Value = breaker opening time in ms for `Trip`).
- **Findings:** `Relay ID, Date, Event`.
""")
            dm = demo_full()
            for col, kk in zip(st.columns(4), NEED):
                col.download_button(kk.capitalize(), dm[kk].to_csv(index=False).encode('utf-8-sig'), f"demo_relay_{kk}.csv", 'text/csv', key=f"dl_{kk}")
    h1, h2 = st.columns(2)
    thr = h1.number_input("Alert threshold (%)", 50, 95, 70, 5)
    lb = h2.number_input("Look-back window (days)", 30, 1095, 365, 30)
    if st.button("▶️ Run Historical Study", type='primary', key='h_go'):
        data, rows = None, None
        if hsrc.startswith("Demo"):
            data = demo_full(); rows = demo_series()
        else:
            for e in errs:
                st.error(e)
            if not errs and all(files.get(k) is not None for k in ['register', 'tests', 'events']):
                data = files
                rows = run_series(files['register'], files['tests'], files['events'])
            elif not errs:
                st.error("Load the relay register, the OMICRON test points and the event log, plus the findings file.")
        if rows:
            tl, evs, fa, n_ep = rc.historical_study(rows, data.get('findings'), thr, lb)
            modes = pd.DataFrame([dict(**{'Relay ID': a['Relay ID'], 'Date': pd.Timestamp(a['Timestamp']).floor('D')}, **a['Mode Risks']) for a in rows])
            st.session_state['hist'] = dict(tl=tl, evs=evs, fa=fa, n_ep=n_ep, thr=thr, events=data.get('findings'), modes=modes,
                                            demo=hsrc.startswith("Demo"))
    H = st.session_state['hist']
    if H is not None:
        tl, evs = H['tl'], H['evs']
        if H['demo']:
            st.info("Synthetic demo data. Not real measurements.")
        caught = int((evs['Caught'] == 'Yes').sum()) if len(evs) else 0
        lt = evs['Lead Time (days)'].dropna() if len(evs) else pd.Series(dtype=float)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Relays", tl['Relay ID'].nunique())
        k2.metric("Findings caught", f"{caught} / {len(evs)}" if len(evs) else "no findings")
        k3.metric("Median warning before finding", f"{int(lt.median())} days" if len(lt) else "—")
        k4.metric("Alert episodes with no finding", f"{H['fa']} of {H['n_ep']}")
        show = tl.groupby('Relay ID')['Overall Risk (%)'].max()
        show = show[show > rc.EXCELLENT_MAX].index.tolist()
        pal = ['#2a78d6', '#eb6834', '#1baf7a', '#8e5cd6', '#e87ba4', '#555555', '#eda100', '#008300']
        fig = go.Figure()
        for i, rid in enumerate(show):
            g = tl[tl['Relay ID'] == rid]
            fig.add_scatter(x=g['Date'], y=g['Overall Risk (%)'], mode='lines', name=rid, line=dict(color=pal[i % len(pal)]))
        fig.add_hline(y=H['thr'], line_dash='dash', line_color=RED, annotation_text=f"Alert threshold {H['thr']}%", annotation_position='top left')
        if H['events'] is not None:
            for _, e in H['events'].iterrows():
                d_ = pd.to_datetime(e['Date'], errors='coerce')
                if pd.notna(d_):
                    fig.add_vline(x=d_, line_dash='dot', line_color='#555')
                    fig.add_annotation(x=d_, y=104, text=f"⚠ {e['Relay ID']}", showarrow=False)
        fig.update_layout(title='Failure risk over time (relays that ever left the normal band)', height=440,
                          yaxis=dict(range=[0, 110], title='Failure Risk (%)'), legend=dict(orientation='h', y=-0.2), margin=dict(l=50, r=20, t=50, b=40))
        st.plotly_chart(fig, width=STRETCH, key='h_fig')
        if len(evs):
            st.markdown("#### Findings vs. warnings")
            st.dataframe(evs, hide_index=True, width=STRETCH)
        st.markdown("#### One relay in detail")
        rids = sorted(tl['Relay ID'].unique())
        sel = st.selectbox("Relay", rids, index=rids.index(show[0]) if show else 0, key='h_sel')
        g = H['modes'][H['modes']['Relay ID'] == sel]
        fig2 = go.Figure()
        for i, m in enumerate(MODE_ORDER):
            if g[m].max() > 5:
                fig2.add_scatter(x=g['Date'], y=g[m], mode='lines', name=m, line=dict(color=pal[i % len(pal)]))
        fig2.add_hline(y=35, line_dash='dash', line_color=AMBER); fig2.add_hline(y=70, line_dash='dash', line_color=RED)
        if H['events'] is not None:
            for _, e in H['events'][H['events']['Relay ID'].astype(str) == sel].iterrows():
                fig2.add_vline(x=pd.to_datetime(e['Date']), line_dash='dot', line_color='#555')
                fig2.add_annotation(x=pd.to_datetime(e['Date']), y=104, text=f"⚠ {str(e['Event'])[:45]}", showarrow=False)
        fig2.update_layout(title=f'{sel}: risk by failure mode over time', height=380, yaxis=dict(range=[0, 110], title='Risk (%)'),
                           legend=dict(orientation='h', y=-0.2), margin=dict(l=50, r=20, t=50, b=40))
        st.plotly_chart(fig2, width=STRETCH, key='h_fig2')
        st.caption("'Alert episodes with no finding' counts alert periods (alerts less than 120 days apart form one episode) "
                   "not followed by a recorded finding within the look-back window.")
        st.download_button("📥 Download daily relay timeline (CSV)", tl.to_csv(index=False).encode('utf-8-sig'),
                           'relayguard_historical_timeline.csv', 'text/csv')
    with st.expander("How the failure risk is calculated"):
        st.markdown("""
Each relay gets a risk (0-100%) for seven failure modes:

| Failure mode | Main inputs | Rule |
|---|---|---|
| Measurement and timing | OMICRON test points: error as % of tolerance, drift since the previous test, mal-operations | ≤ 50% of tolerance normal; outside tolerance ≥ 70%; drift ≥ 30% of tolerance adds 10-20; a mal-operation ≥ 85% |
| Breaker | opening time of the last trips against rated, its trend, trips per year, breaker failure | 1.1× rated starts the risk, 1.5× = 90% |
| Secondary circuits | trip circuit, CT and VT supervision alarms (30 days) | trip circuit ≥ 75%; CT ≥ 60%; VT fuse failure ≥ 55% when distance/directional is used |
| Hardware | self-test alarms, watchdog, power supply, restarts (90 days) | watchdog / relay failure ≥ 85% |
| Communication | communication failures, time sync losses (30 days) | +15 on a line differential relay (protection signalling) |
| Age and obsolescence | age against expected life (numerical 20, static 25, electromechanical 30 years), firmware, self-supervision | |
| Test overdue | months since the last test against the test interval | 1.5× the interval = 60% |

A `Maintenance` entry in the event log (repair completed) resets the alarm windows and the breaker timing history.
**Overall risk** = the highest mode risk + 5 points for every other mode above normal. Bands: ≤ 35% normal, ≤ 70% watch, > 70% action.
**FMEA:** RPN = S × O × D with S per failure mode, O = risk ÷ 10, D = 2.
""")

st.markdown(FOOTER, unsafe_allow_html=True)
