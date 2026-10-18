"""
RelayGuard AI - condition and risk engine for protection relays and their breakers.

One engine used by every page of the app, the plant analysis and the historical study.

Inputs (CSV):
  * Relay register : Relay ID, Bay, Protected Object, Role, Functions, Technology, Commission Year,
                     Firmware, Latest Firmware, Test Interval (months), Breaker ID, CB Rated Opening Time (ms)
  * OMICRON tests  : Date, Relay ID, Function, Test Point, Quantity, Unit, Nominal, Measured, Tolerance, Result
                     (one row per test point, flattened from the test-set report)
  * Event log (SER): Timestamp, Relay ID, Event, Value     (Value = breaker opening time in ms for 'Trip')
  * Findings       : Relay ID, Date, Event                  (failures and findings, for the historical study)
"""
import math
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- vocabulary and limits
EVENT_TYPES = ['Self-test alarm', 'Relay failure', 'Power supply alarm', 'Reboot', 'Comms failure', 'Time sync loss',
               'CT supervision', 'VT fuse failure', 'Trip circuit supervision', 'Trip', 'Breaker failure', 'Mal-operation',
               'Settings change', 'Maintenance']
# 'Maintenance' = repair or replacement completed on the relay or its breaker: older alarms and trip times stop counting
WIN_HW, WIN_COMMS, WIN_SEC, WIN_YEAR, WIN_CB = 90, 30, 30, 365, 730
EXPECTED_LIFE = {'Numerical': 20.0, 'Static': 25.0, 'Electromechanical': 30.0}
EXCELLENT_MAX, WATCH_MAX = 35.0, 70.0
DETECTION = 2

MODES = {
    'Measurement and timing': {'S': 9, 'text': 'Relay measures or times wrongly: wrong zone, slow or missed trip'},
    'Breaker': {'S': 9, 'text': 'Breaker opens slowly or fails to open'},
    'Secondary circuits': {'S': 8, 'text': 'CT, VT or trip circuit problem: protection blind or trip cannot reach the breaker'},
    'Hardware': {'S': 8, 'text': 'Relay hardware failing (self-supervision alarms, power supply, restarts)'},
    'Communication': {'S': 6, 'text': 'Protection signalling, station bus or time synchronisation problems'},
    'Age and obsolescence': {'S': 5, 'text': 'Relay near the end of its life, old firmware, no self-supervision'},
    'Test overdue': {'S': 4, 'text': 'Periodic secondary injection test overdue'},
    'Record anomaly (AI)': {'S': 7, 'text': 'Disturbance records unlike the normal fleet behaviour learned by the AI model (analog input noise, measurement chain)'},
}
MODE_ORDER = list(MODES)
ACTIONS = {
    'Measurement and timing': 'Repeat the secondary injection test on the affected functions; recalibrate or replace the relay; review the last fault records.',
    'Breaker': 'Plan breaker timing test and mechanism overhaul (trip coil, latch, lubrication); check the trip circuit.',
    'Secondary circuits': 'Inspect CT / VT secondary wiring, terminals and fuses, and the trip circuit, before the next fault.',
    'Hardware': 'Read the relay self-supervision log; plan replacement of the faulty module or the relay; keep a spare ready.',
    'Communication': 'Check the protection signalling channel (fibre, multiplexer), the station bus and the time server.',
    'Age and obsolescence': 'Plan relay replacement in the capital programme; update firmware where the vendor recommends.',
    'Test overdue': 'Schedule the periodic secondary injection test.',
    'Record anomaly (AI)': 'Review the latest disturbance records; check the analog input module, CT/VT wiring and earthing; compare with a secondary injection test.',
}


# ---------------------------------------------------------------- AI: disturbance-record anomaly model
WIN_REC = 365
AI_FULL = 99.5   # percentile of the fleet at which the record anomaly risk is full
AI_START = 90.0


def record_matrix(current, noise):
    """Model inputs from the disturbance-record features: fault current, pre-fault noise and their ratio (log scale)."""
    c = np.clip(np.asarray(current, float), 0.01, None); n = np.clip(np.asarray(noise, float), 1e-4, None)
    return np.c_[np.log10(c), np.log10(n), np.log10(n / c)]


def load_record_model(path='record_anomaly_model.pkl'):
    import pickle
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None


def ai_percentile(bundle, current, noise):
    """How unusual each record is compared with the records the model learned from (0-100; 99 = more unusual than 99%)."""
    if bundle is None:
        return np.full(len(np.atleast_1d(current)), np.nan)
    s = -bundle['model'].score_samples(record_matrix(np.atleast_1d(current), np.atleast_1d(noise)))
    return 100.0 * np.searchsorted(bundle['train_scores'], s, side='right') / len(bundle['train_scores'])


def clamp(x, lo=0.0, hi=100.0):
    return float(min(hi, max(lo, x)))


def lin(x, x0, x1, y0=0.0, y1=100.0):
    if x1 == x0:
        return y1 if x >= x1 else y0
    return clamp(y0 + (y1 - y0) * (x - x0) / (x1 - x0), min(y0, y1), max(y0, y1))


def status_of(r):
    return 'EXCELLENT' if r <= EXCELLENT_MAX else ('WATCH' if r <= WATCH_MAX else 'ACTION')


def rpn_level(rpn):
    return 'Low' if rpn < 40 else ('Medium' if rpn < 100 else 'High')


def isnum(v):
    try:
        return v is not None and v != '' and not (isinstance(v, float) and math.isnan(v)) and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def reg_value(reg, key, default):
    v = reg.get(key, default)
    if v is None or v == '' or (isinstance(v, float) and math.isnan(v)):
        return default
    return v


def technology(reg):
    t = str(reg_value(reg, 'Technology', 'Numerical')).strip().lower()
    return 'Electromechanical' if t.startswith('elec') else 'Static' if t.startswith('stat') else 'Numerical'


# ---------------------------------------------------------------- test points
def point_error(p):
    """Error of one test point as a fraction of its tolerance (1.0 = at the tolerance limit)."""
    nom, mea, tol = p.get('Nominal'), p.get('Measured'), p.get('Tolerance')
    if not (isnum(nom) and isnum(mea) and isnum(tol)) or float(tol) <= 0:
        return None
    e = abs(float(mea) - float(nom)) / float(tol)
    if str(p.get('Result', '')).strip().lower().startswith('fail'):
        e = max(e, 1.01)
    return e


def point_score(e):
    if e <= 0.5:
        return lin(e, 0, 0.5, 0, 10)
    if e <= 1.0:
        return lin(e, 0.5, 1.0, 10, 50)
    return lin(e, 1.0, 2.0, 70, 100)


def point_name(p):
    return ' '.join(str(p.get(k, '')).strip() for k in ['Function', 'Test Point', 'Quantity'] if str(p.get(k, '')).strip())


def fmt_num(v):
    v = float(v)
    return f"{v:.0f}" if abs(v) >= 1000 else f"{v:.4g}"


# ---------------------------------------------------------------- one relay
def assess(reg, s, today=None):
    """Assess one relay.
    reg: register fields.
    s: summary of the evidence on the assessment date:
       'points'      list of test points of the latest test session (Function, Test Point, Quantity, Unit, Nominal,
                     Measured, Tolerance, Result, optional Previous Measured)
       'last_test'   date of the latest test (Timestamp or None)
       counts        'Self-test alarm', 'Relay failure', 'Power supply alarm', 'Reboot' (last 90 days),
                     'Comms failure', 'Time sync loss', 'CT supervision', 'VT fuse failure', 'Trip circuit supervision' (30 days),
                     'Mal-operation', 'Breaker failure', 'Trips' (365 days)
       'cb_times'    breaker opening times (ms) of the trips in the last 2 years, oldest first
    today: assessment date (default now)."""
    today = pd.Timestamp(today if today is not None else pd.Timestamp.now()).normalize()
    c = lambda k: int(s.get(k, 0) or 0)
    funcs = str(reg_value(reg, 'Functions', ''))
    tech = technology(reg)
    out = {'Relay ID': str(reg_value(reg, 'Relay ID', '')), 'Bay': reg_value(reg, 'Bay', ''),
           'Protected Object': reg_value(reg, 'Protected Object', ''), 'Role': reg_value(reg, 'Role', ''),
           'Functions': funcs, 'Technology': tech, 'Breaker ID': reg_value(reg, 'Breaker ID', ''), 'Assessed On': today}
    R, why = {}, {}

    # --- measurement and timing (OMICRON)
    pts = []
    for p in s.get('points') or []:
        e = point_error(p)
        if e is None:
            continue
        prev = None
        if isnum(p.get('Previous Measured')) and isnum(p.get('Nominal')) and isnum(p.get('Tolerance')) and float(p['Tolerance']) > 0:
            prev = abs(float(p['Previous Measured']) - float(p['Nominal'])) / float(p['Tolerance'])
        pts.append({**p, 'Error (% of tol)': e * 100, 'Pass': e <= 1.0, 'Drift (% of tol)': None if prev is None else (e - prev) * 100,
                    'Score': point_score(e)})
    w = []
    if pts:
        worst = max(pts, key=lambda p: p['Score'])
        s_m = worst['Score']
        fails = [p for p in pts if not p['Pass']]
        w.append(f"{len(pts)} test points, {len(fails)} outside tolerance. Worst: {point_name(worst)} "
                 f"{fmt_num(worst['Measured'])} against {fmt_num(worst['Nominal'])} ± {fmt_num(worst['Tolerance'])} {worst.get('Unit', '') or ''}"
                 f" ({worst['Error (% of tol)']:.0f}% of tolerance).".replace('  ', ' '))
        if len(fails) > 1:
            s_m += min(15, 5 * (len(fails) - 1))
        dr = [p for p in pts if p['Drift (% of tol)'] is not None]
        if dr:
            d = max(dr, key=lambda p: p['Drift (% of tol)'])
            if d['Drift (% of tol)'] >= 30:
                s_m += lin(d['Drift (% of tol)'], 30, 80, 10, 20)
                w.append(f"{point_name(d)} drifted {d['Drift (% of tol)']:.0f}% of tolerance since the previous test.")
    else:
        s_m = 0.0
        w.append('No secondary injection test results.')
    if c('Mal-operation'):
        s_m = max(s_m, 85); w.append(f"{c('Mal-operation')} mal-operation(s) recorded in the last year.")
    R['Measurement and timing'], why['Measurement and timing'] = clamp(s_m), w
    out['Test Points'] = pts

    # --- breaker
    times = [float(t) for t in (s.get('cb_times') or []) if isnum(t) and float(t) > 0]
    rated = float(reg_value(reg, 'CB Rated Opening Time (ms)', 0) or 0)
    w, s_b = [], 0.0
    out['CB Opening Time (ms)'] = None
    if str(out['Breaker ID']).strip() and rated > 0:
        if times:
            last3 = times[-3:]
            med = float(np.median(last3))
            ratio = med / rated
            out['CB Opening Time (ms)'] = med
            s_b = lin(ratio, 1.1, 1.5, 0, 90)
            w.append(f"Breaker {out['Breaker ID']}: opening time {med:.0f} ms (median of the last {len(last3)} trips) against {rated:.0f} ms rated ({ratio:.2f}×).")
            if len(times) >= 4:
                y = np.array(times[-6:]); x = np.arange(len(y))
                slope = float(np.polyfit(x, y, 1)[0]) / rated * 100
                if slope > 1:
                    s_b += lin(slope, 1, 5, 0, 20)
                    w.append(f"Opening time rising {slope:.1f}% of rated per operation.")
        else:
            w.append(f"Breaker {out['Breaker ID']}: no trip recorded in the last 2 years.")
        if c('Trips') > 15:
            s_b += lin(c('Trips'), 15, 40, 0, 20); w.append(f"{c('Trips')} trips in the last year (mechanism wear).")
        if c('Breaker failure'):
            s_b = max(s_b, 85); w.append(f"Breaker failure protection operated {c('Breaker failure')} time(s) in the last year.")
    else:
        w.append('No breaker assigned to this relay.')
    R['Breaker'], why['Breaker'] = clamp(s_b), w

    # --- secondary circuits
    parts, w = [], []
    if c('Trip circuit supervision'):
        parts.append(75 + 5 * (c('Trip circuit supervision') - 1)); w.append(f"Trip circuit supervision alarm {c('Trip circuit supervision')} time(s) in 30 days: the trip may not reach the breaker.")
    if c('CT supervision'):
        parts.append(60 + 5 * (c('CT supervision') - 1)); w.append(f"CT supervision alarm {c('CT supervision')} time(s) in 30 days.")
    if c('VT fuse failure'):
        base = 55 if ('21' in funcs or '67' in funcs or '27' in funcs) else 40
        parts.append(base + 5 * (c('VT fuse failure') - 1)); w.append(f"VT fuse failure {c('VT fuse failure')} time(s) in 30 days" + (": distance / directional functions blocked." if base == 55 else "."))
    R['Secondary circuits'] = clamp(max(parts) if parts else 0.0)
    why['Secondary circuits'] = w or ['No CT, VT or trip circuit alarms in the last 30 days.']

    # --- hardware
    parts, w = [], []
    if c('Relay failure'):
        parts.append(85 + 5 * (c('Relay failure') - 1)); w.append(f"Relay failure (watchdog) {c('Relay failure')} time(s) in 90 days.")
    if c('Self-test alarm'):
        parts.append(lin(c('Self-test alarm'), 1, 6, 15, 60)); w.append(f"{c('Self-test alarm')} self-test alarm(s) in 90 days.")
    if c('Power supply alarm'):
        parts.append(lin(c('Power supply alarm'), 1, 3, 30, 60)); w.append(f"{c('Power supply alarm')} power supply alarm(s) in 90 days.")
    if c('Reboot'):
        parts.append(lin(c('Reboot'), 1, 4, 20, 60)); w.append(f"{c('Reboot')} restart(s) in 90 days.")
    s_h = max(parts) if parts else 0.0
    if len(parts) >= 2:
        s_h += 10
    R['Hardware'] = clamp(s_h)
    why['Hardware'] = w or ['No self-supervision alarms in the last 90 days.']

    # --- communication
    parts, w = [], []
    if c('Comms failure'):
        v = lin(c('Comms failure'), 1, 8, 15, 70)
        if '87L' in funcs:
            v += 15; w.append(f"{c('Comms failure')} communication failure(s) in 30 days on a line differential relay (protection signalling).")
        else:
            w.append(f"{c('Comms failure')} communication failure(s) in 30 days.")
        parts.append(v)
    if c('Time sync loss'):
        parts.append(lin(c('Time sync loss'), 1, 8, 10, 45)); w.append(f"{c('Time sync loss')} time synchronisation loss(es) in 30 days.")
    s_c = max(parts) if parts else 0.0
    if len(parts) == 2:
        s_c += 10
    R['Communication'] = clamp(s_c)
    why['Communication'] = w or ['No communication or time sync alarms in the last 30 days.']

    # --- age and obsolescence
    yr = reg_value(reg, 'Commission Year', None)
    life = EXPECTED_LIFE[tech]
    w = []
    if isnum(yr):
        age = today.year - int(float(yr))
        s_a = lin(age / life, 0.7, 1.3, 0, 75)
        w.append(f"{tech} relay, {age} years in service against about {life:.0f} years expected life.")
    else:
        age, s_a = None, 0.0
        w.append('Commission year not given.')
    fw, lfw = str(reg_value(reg, 'Firmware', '')).strip(), str(reg_value(reg, 'Latest Firmware', '')).strip()
    if fw and lfw and fw != lfw:
        s_a += 10; w.append(f"Firmware {fw}, latest {lfw}.")
    if tech != 'Numerical':
        s_a += 10; w.append('No continuous self-supervision: hidden failures are found only at the periodic test.')
    out['Age (years)'] = age
    R['Age and obsolescence'], why['Age and obsolescence'] = clamp(s_a), w

    # --- test overdue
    interval = float(reg_value(reg, 'Test Interval (months)', 24) or 24)
    lt = s.get('last_test')
    if lt is not None and not pd.isna(lt):
        months = (today - pd.Timestamp(lt).normalize()).days / 30.4375
        ratio = months / interval if interval > 0 else 0
        s_t = lin(ratio, 1.0, 1.5, 0, 60) if ratio <= 1.5 else lin(ratio, 1.5, 2.5, 60, 75)
        w = [f"Last test {pd.Timestamp(lt).date()} ({months:.0f} months ago, interval {interval:.0f} months)."]
        out['Months Since Test'] = months
    else:
        s_t = 70.0; w = ['No test on record.']
        out['Months Since Test'] = None
    R['Test overdue'], why['Test overdue'] = s_t, w

    # --- AI: disturbance-record anomaly
    recs = [r for r in (s.get('records') or []) if isnum(r.get('AI Percentile'))]
    w = []
    if recs:
        last = recs[-3:]
        med = float(np.median([r['AI Percentile'] for r in last]))
        s_ai = lin(med, AI_START, AI_FULL, 0, 90)
        top = max(last, key=lambda r: r['AI Percentile'])
        w.append(f"AI model: the last {len(last)} disturbance record(s) are more unusual than {med:.0f}% of the real records it learned from "
                 f"(most unusual: noise index {fmt_num(top['Noise Index'])} at {fmt_num(top['Max Fault Current (A)'])} A).")
        out['AI Percentile'] = med
    else:
        s_ai = 0.0; w.append('No disturbance records in the last year for the AI model.')
        out['AI Percentile'] = None
    R['Record anomaly (AI)'], why['Record anomaly (AI)'] = clamp(s_ai), w

    # --- overall
    risks = {k: round(R[k], 1) for k in MODE_ORDER}
    gov = max(risks, key=risks.get)
    others = sum(1 for k, v in risks.items() if k != gov and v > EXCELLENT_MAX)
    overall = clamp(risks[gov] + 5 * others)
    S = MODES[gov]['S']; O = max(1, math.ceil(overall / 10)); rpn = S * O * DETECTION
    fmea = []
    for k, v in sorted(risks.items(), key=lambda kv: -kv[1]):
        o = max(1, math.ceil(v / 10))
        fmea.append({'Failure Mode': k, 'Risk (%)': v, 'S': MODES[k]['S'], 'O': o, 'D': DETECTION,
                     'RPN': MODES[k]['S'] * o * DETECTION, 'RPN Level': rpn_level(MODES[k]['S'] * o * DETECTION)})
    st = status_of(round(overall, 1))
    out.update({'Mode Risks': risks, 'Reasons': why, 'Governing Mode': gov, 'Other Modes Above Normal': others,
                'Overall Risk (%)': round(overall, 1), 'Status': st, 'S': S, 'O': O, 'D': DETECTION, 'RPN': rpn,
                'RPN Level': rpn_level(rpn), 'FMEA': fmea,
                'Action': ACTIONS[gov] if st != 'EXCELLENT' else 'Continue monitoring and periodic testing.'})
    return out


def findings(a):
    r = []
    for k, v in sorted(a['Mode Risks'].items(), key=lambda kv: -kv[1]):
        if k == a['Governing Mode'] or v > EXCELLENT_MAX:
            r.append(f"{k} ({v:.0f}%): " + ' '.join(a['Reasons'][k]))
    n = a['Other Modes Above Normal']
    if n:
        r.append(f"{n} other failure mode(s) above normal add {5 * n} points to the overall risk.")
    if a['Status'] == 'EXCELLENT':
        r.append('All failure modes within normal limits.')
    return r


# ---------------------------------------------------------------- evidence from the files
def prepare(register, tests=None, events=None, records=None, model=None):
    """Index the files once; summary() then builds the evidence of any relay on any date."""
    reg = {str(r['Relay ID']): r for r in register.to_dict('records')}
    T = {}
    if tests is not None and len(tests):
        t = tests.copy()
        t['_d'] = pd.to_datetime(t['Date'], errors='coerce').dt.normalize()
        t['Relay ID'] = t['Relay ID'].astype(str)
        t = t.dropna(subset=['_d'])
        for c in ['Nominal', 'Measured', 'Tolerance']:
            t[c] = pd.to_numeric(t[c], errors='coerce')
        for rid, g in t.groupby('Relay ID'):
            sess = {d: gg.drop(columns=['_d']).to_dict('records') for d, gg in g.groupby('_d')}
            T[rid] = (sorted(sess), sess)
    E = {}
    if events is not None and len(events):
        e = events.copy()
        e['_t'] = pd.to_datetime(e['Timestamp'], errors='coerce')
        e = e.dropna(subset=['_t'])
        e['Relay ID'] = e['Relay ID'].astype(str); e['Event'] = e['Event'].astype(str).str.strip()
        e['Value'] = pd.to_numeric(e.get('Value'), errors='coerce')
        e = e.sort_values('_t')
        for (rid, ev), g in e.groupby(['Relay ID', 'Event']):
            E.setdefault(rid, {})[ev] = (g['_t'].to_numpy(dtype='datetime64[ns]'), g['Value'].to_numpy())
    RC = {}
    if records is not None and len(records):
        r = records.copy()
        r['_t'] = pd.to_datetime(r['Timestamp'], errors='coerce'); r = r.dropna(subset=['_t'])
        r['Relay ID'] = r['Relay ID'].astype(str)
        for c in ['Max Fault Current (A)', 'Noise Index']:
            r[c] = pd.to_numeric(r[c], errors='coerce')
        r = r.dropna(subset=['Max Fault Current (A)', 'Noise Index']).sort_values('_t')
        if len(r):
            r['AI Percentile'] = ai_percentile(model, r['Max Fault Current (A)'].to_numpy(), r['Noise Index'].to_numpy())
        for rid, g in r.groupby('Relay ID'):
            g = g.assign(Date=g['_t'].dt.strftime('%Y-%m-%d'))
            RC[rid] = (g['_t'].to_numpy(dtype='datetime64[ns]'), g[['Date', 'Max Fault Current (A)', 'Noise Index', 'AI Percentile']].to_dict('records'))
    return {'reg': reg, 'tests': T, 'events': E, 'records': RC}


def summary(P, rid, day):
    """Evidence for relay rid at the end of day (only data up to that day)."""
    day = pd.Timestamp(day).normalize()
    end = np.datetime64(day + pd.Timedelta(days=1), 'ns')
    s = {'points': [], 'last_test': None, 'cb_times': []}
    if rid in P['tests']:
        dates, sess = P['tests'][rid]
        i = bisect_right(dates, day)
        if i:
            cur = dates[i - 1]
            s['last_test'] = cur
            prev = {}
            if i >= 2:
                for p in sess[dates[i - 2]]:
                    prev[(p.get('Function'), p.get('Test Point'), p.get('Quantity'))] = p.get('Measured')
            for p in sess[cur]:
                q = dict(p)
                q['Previous Measured'] = prev.get((p.get('Function'), p.get('Test Point'), p.get('Quantity')))
                s['points'].append(q)
    ev = P['events'].get(rid, {})
    reset = None
    if 'Maintenance' in ev:
        mt = ev['Maintenance'][0]
        j = int(np.searchsorted(mt, end, 'left'))
        if j:
            reset = mt[j - 1]
            s['last_maintenance'] = pd.Timestamp(reset)
    for k, win in [('Self-test alarm', WIN_HW), ('Relay failure', WIN_HW), ('Power supply alarm', WIN_HW), ('Reboot', WIN_HW),
                   ('Comms failure', WIN_COMMS), ('Time sync loss', WIN_COMMS), ('CT supervision', WIN_SEC), ('VT fuse failure', WIN_SEC),
                   ('Trip circuit supervision', WIN_SEC), ('Mal-operation', WIN_YEAR), ('Breaker failure', WIN_YEAR), ('Trip', WIN_YEAR)]:
        if k in ev:
            ts = ev[k][0]
            lo = end - np.timedelta64(win, 'D')
            if reset is not None and reset > lo:
                lo = reset
            n = int(np.searchsorted(ts, end, 'left') - np.searchsorted(ts, lo, 'left'))
        else:
            n = 0
        s['Trips' if k == 'Trip' else k] = n
    if 'Trip' in ev:
        ts, vals = ev['Trip']
        lo = end - np.timedelta64(WIN_CB, 'D')
        if reset is not None and reset > lo:
            lo = reset
        a = int(np.searchsorted(ts, lo, 'left')); b = int(np.searchsorted(ts, end, 'left'))
        s['cb_times'] = [float(v) for v in vals[a:b] if not np.isnan(v)]
    if rid in P.get('records', {}):
        ts, recs = P['records'][rid]
        lo = end - np.timedelta64(WIN_REC, 'D')
        if reset is not None and reset > lo:
            lo = reset
        a = int(np.searchsorted(ts, lo, 'left')); b = int(np.searchsorted(ts, end, 'left'))
        s['records'] = recs[a:b]
    return s


def assess_files(register, tests=None, events=None, day=None, records=None, model=None):
    """Latest assessment of every relay in the register."""
    P = prepare(register, tests, events, records, model)
    if day is None:
        cands = [pd.Timestamp.now().normalize()]
        day = cands[0]
    return [assess(r, summary(P, rid, day), day) for rid, r in P['reg'].items()]


def assess_series(register, tests=None, events=None, start=None, end=None, records=None, model=None):
    """Daily assessment of every relay from start to end, each day with only the data up to that day."""
    P = prepare(register, tests, events, records, model)
    if start is None or end is None:
        ts = []
        for rid, d in P['events'].items():
            for k, (t, _) in d.items():
                ts += [t.min(), t.max()]
        start = start or (pd.Timestamp(min(ts)).normalize() if ts else pd.Timestamp.now().normalize())
        end = end or (pd.Timestamp(max(ts)).normalize() if ts else pd.Timestamp.now().normalize())
    rows = []
    for day in pd.date_range(start, end, freq='D'):
        for rid, r in P['reg'].items():
            a = assess(r, summary(P, rid, day), day)
            a['Timestamp'] = day
            rows.append(a)
    return rows


def latest(rows):
    last = {}
    for a in rows:
        if a['Relay ID'] not in last or a['Timestamp'] >= last[a['Relay ID']]['Timestamp']:
            last[a['Relay ID']] = a
    return sorted(last.values(), key=lambda a: -a['Overall Risk (%)'])


def historical_study(rows, events=None, alert_threshold=70.0, lookback_days=365, gap_days=120):
    tl = pd.DataFrame([{'Relay ID': a['Relay ID'], 'Date': pd.Timestamp(a['Timestamp']).floor('D'), 'Overall Risk (%)': a['Overall Risk (%)'],
                        'Status': a['Status'], 'Governing Mode': a['Governing Mode']} for a in rows])
    tl = tl.sort_values(['Relay ID', 'Date']).groupby(['Relay ID', 'Date'], as_index=False).last()
    episodes = []
    for m, g in tl[tl['Overall Risk (%)'] >= alert_threshold].groupby('Relay ID'):
        cur = None
        for d in g['Date']:
            if cur is not None and (d - cur[2]).days <= gap_days:
                cur[2] = d
            else:
                if cur is not None:
                    episodes.append(cur)
                cur = [m, d, d]
        if cur is not None:
            episodes.append(cur)
    ev = pd.DataFrame(columns=['Relay ID', 'Date', 'Event']) if events is None else events.copy()
    ev['_d'] = pd.to_datetime(ev['Date'], errors='coerce'); ev = ev.dropna(subset=['_d']); ev['Relay ID'] = ev['Relay ID'].astype(str)
    out = []
    for _, e in ev.iterrows():
        prior = [p for p in episodes if p[0] == e['Relay ID'] and p[1] < e['_d'] and (e['_d'] - p[2]).days <= lookback_days]
        first = min(p[1] for p in prior) if prior else None
        gov = tl[(tl['Relay ID'] == e['Relay ID']) & (tl['Date'] == first)] if first is not None else None
        out.append({'Relay ID': e['Relay ID'], 'Event Date': e['_d'].date(), 'Event': e.get('Event', ''),
                    'First Alert Before Event': first.date() if first is not None else None,
                    'Lead Time (days)': (e['_d'] - first).days if first is not None else None,
                    'Mode at Alert': gov['Governing Mode'].iloc[0] if gov is not None and len(gov) else None,
                    'Caught': 'Yes' if first is not None else 'No'})
    fa = sum(1 for m, st, en in episodes if not ((ev['Relay ID'] == m) & (ev['_d'] >= st) & (ev['_d'] <= en + pd.Timedelta(days=lookback_days))).any())
    return tl, pd.DataFrame(out), fa, len(episodes)


# ---------------------------------------------------------------- demo substation
TEST_PLANS = {
    '21': [('21', 'Zone 1', 'Reach', 'Ω', 8.0, 0.4), ('21', 'Zone 2', 'Reach', 'Ω', 12.0, 0.6), ('21', 'Zone 1', 'Trip time', 'ms', 30.0, 15.0),
           ('21', 'Zone 2', 'Trip time', 'ms', 400.0, 20.0), ('67N', 'Forward', 'Pickup', 'A', 0.20, 0.01)],
    '87L': [('87L', 'Differential', 'Pickup', 'A', 0.30, 0.015), ('87L', 'Slope 1', 'Bias', '%', 30.0, 1.5), ('87L', 'Differential', 'Trip time', 'ms', 30.0, 10.0)],
    '87T': [('87T', 'Differential', 'Pickup', 'A', 0.30, 0.015), ('87T', '2nd harmonic', 'Block', '%', 15.0, 1.5), ('87T', 'Differential', 'Trip time', 'ms', 30.0, 10.0)],
    '51': [('51', 'IDMT', 'Pickup', 'A', 1.0, 0.05), ('51', 'IDMT at 5× pickup', 'Trip time', 'ms', 1200.0, 60.0), ('50', 'High set', 'Pickup', 'A', 10.0, 0.5)],
    '87B': [('87B', 'Differential', 'Pickup', 'A', 0.50, 0.025), ('87B', 'Differential', 'Trip time', 'ms', 20.0, 10.0)],
    'EM': [('87R', 'Differential', 'Pickup', 'A', 0.20, 0.02), ('87R', 'Differential', 'Trip time', 'ms', 40.0, 15.0)],
}


def demo_register():
    rows = [
        ['L1-M1', 'Line 1', 'Line', 'Main 1', '21, 67N, 79', 'Numerical', 2016, '7.4', '7.4', 24, 'Q1-CB1', 40, '21'],
        ['L1-M2', 'Line 1', 'Line', 'Main 2', '87L, 21', 'Numerical', 2016, '5.2', '5.2', 24, '', 0, '87L'],
        ['L2-M1', 'Line 2', 'Line', 'Main 1', '21, 67N, 79', 'Numerical', 2012, '7.1', '7.4', 24, 'Q2-CB1', 40, '21'],
        ['L2-M2', 'Line 2', 'Line', 'Main 2', '87L, 21', 'Numerical', 2012, '5.0', '5.2', 24, '', 0, '87L'],
        ['L3-M1', 'Line 3', 'Line', 'Main 1', '21, 67N, 79', 'Numerical', 2019, '7.4', '7.4', 24, 'Q3-CB1', 40, '21'],
        ['L3-M2', 'Line 3', 'Line', 'Main 2', '87L, 21', 'Numerical', 2019, '5.2', '5.2', 24, '', 0, '87L'],
        ['L4-M1', 'Line 4', 'Line', 'Main 1', '21, 67N, 79', 'Numerical', 2010, '6.8', '7.4', 24, 'Q4-CB1', 40, '21'],
        ['L4-M2', 'Line 4', 'Line', 'Main 2', '87L, 21', 'Numerical', 2010, '4.6', '5.2', 24, '', 0, '87L'],
        ['T1-87', 'Transformer 1', 'Transformer', 'Main', '87T, REF', 'Numerical', 2014, '3.3', '3.3', 24, 'T1-CB', 45, '87T'],
        ['T1-51', 'Transformer 1', 'Transformer', 'Backup', '50/51, 50N/51N', 'Numerical', 2014, '2.9', '2.9', 24, '', 0, '51'],
        ['T2-87', 'Transformer 2', 'Transformer', 'Main', '87T, REF', 'Numerical', 2018, '3.3', '3.3', 24, 'T2-CB', 45, '87T'],
        ['T2-51', 'Transformer 2', 'Transformer', 'Backup', '50/51, 50N/51N', 'Static', 1998, '', '', 12, '', 0, '51'],
        ['BB-87A', 'Busbar A', 'Busbar', 'Main', '87B, 50BF', 'Numerical', 2015, '2.2', '2.2', 24, '', 0, '87B'],
        ['BB-87B', 'Busbar B', 'Busbar', 'Main', '87B, 50BF', 'Numerical', 2015, '2.2', '2.2', 24, '', 0, '87B'],
        ['R1-87', 'Reactor 1', 'Reactor', 'Main', '87R, 50/51', 'Electromechanical', 1996, '', '', 12, 'R1-CB', 50, 'EM'],
    ]
    cols = ['Relay ID', 'Bay', 'Protected Object', 'Role', 'Functions', 'Technology', 'Commission Year', 'Firmware', 'Latest Firmware',
            'Test Interval (months)', 'Breaker ID', 'CB Rated Opening Time (ms)', '_plan']
    return pd.DataFrame(rows, columns=cols)


def demo_data(seed=21, start='2025-01-01', end='2026-08-31'):
    """Synthetic substation history. Scenarios (recorded findings):
       L2-M1  zone-2 timer drifting over three tests -> slow zone-2 trip (mal-operation), timer card replaced
       T1-87  growing self-test alarms and restarts -> power supply card failed, replaced
       L3-M1  breaker Q3-CB1 opening time rising -> mechanism overhaul
       L4-M2  line differential signalling failures -> fibre channel repaired
       BB-87A recurring CT supervision alarms -> loose CT terminal found
       Watch without event: R1-87 (electromechanical, 30 years), T2-51 (static, test overdue)"""
    rng = np.random.default_rng(seed)
    regp = demo_register(); reg = regp.drop(columns=['_plan'])
    D = pd.Timestamp
    S, E = D(start), D(end)
    F = {'L2-M1': D('2026-03-10'), 'T1-87': D('2025-10-05'), 'L3-M1': D('2025-11-20'), 'L4-M2': D('2026-06-15'), 'BB-87A': D('2026-05-20')}

    # ---- OMICRON test sessions
    sched = {'L1-M1': ['2022-03-14', '2024-03-11', '2026-03-09'], 'L1-M2': ['2022-03-15', '2024-03-12', '2026-03-10'],
             'L2-M1': ['2021-04-06', '2023-04-04', '2025-04-08', '2025-10-14', '2026-03-24'], 'L2-M2': ['2021-04-07', '2023-04-05', '2025-04-09'],
             'L3-M1': ['2022-09-20', '2024-09-17'], 'L3-M2': ['2022-09-21', '2024-09-18'],
             'L4-M1': ['2022-11-08', '2024-11-05'], 'L4-M2': ['2022-11-09', '2024-11-06', '2026-06-22'],
             'T1-87': ['2021-12-01', '2023-11-28', '2025-10-08'], 'T1-51': ['2021-12-02', '2023-11-29', '2025-11-25'],
             'T2-87': ['2022-06-07', '2024-06-04'], 'T2-51': ['2022-06-08', '2023-05-16', '2024-04-23', '2025-06-10', '2026-07-14'],
             'BB-87A': ['2023-02-14', '2025-02-11', '2026-05-26'], 'BB-87B': ['2023-02-15', '2025-02-12'],
             'R1-87': ['2023-01-17', '2024-01-23', '2025-01-21', '2026-01-20']}
    trows = []
    for _, r in regp.iterrows():
        for dt in sched[r['Relay ID']]:
            for fn, tp, q, u, nom, tol in TEST_PLANS[r['_plan']]:
                noise = 0.18 if r['Technology'] == 'Numerical' else 0.35
                meas = nom + rng.normal(0, noise * tol / 2)
                if r['Relay ID'] == 'L2-M1' and tp == 'Zone 2' and q == 'Trip time':
                    meas = nom + {'2021-04-06': 2.0, '2023-04-04': 5.0, '2025-04-08': 15.0, '2025-10-14': 27.0, '2026-03-24': 1.0}[dt]
                if r['Relay ID'] == 'T2-51':
                    meas = nom + rng.normal(0, 0.45 * tol / 2)
                dec = 3 if u == 'A' else 2 if u == 'Ω' else 1
                trows.append([dt, r['Relay ID'], fn, tp, q, u, nom, round(meas, dec), tol])
    tests = pd.DataFrame(trows, columns=['Date', 'Relay ID', 'Function', 'Test Point', 'Quantity', 'Unit', 'Nominal', 'Measured', 'Tolerance'])
    tests['Result'] = np.where((tests['Measured'] - tests['Nominal']).abs() <= tests['Tolerance'], 'Pass', 'Fail')

    # ---- event log
    ev = []
    def add(t, rid, e, v=np.nan):
        ev.append([t, rid, e, v])
    days = pd.date_range(S - pd.Timedelta(days=120), E, freq='D')
    num = set(reg[reg['Technology'] == 'Numerical']['Relay ID'])
    for day in days:
        for rid in reg['Relay ID']:
            hh = pd.Timedelta(hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60)))
            if rid in num and rng.random() < 1 / 260:
                add(day + hh, rid, 'Self-test alarm')
            if rid in num and rng.random() < 1 / 300:
                add(day + hh, rid, 'Time sync loss')
            if rid.endswith('M2') and rng.random() < 1 / 300:
                add(day + hh, rid, 'Comms failure')
        # line faults: trips on Main 1 (breaker) and Main 2
        for ln, rate in [('L1', 1 / 45), ('L2', 1 / 50), ('L3', 1 / 14), ('L4', 1 / 40)]:
            if rng.random() < rate:
                t = day + pd.Timedelta(hours=int(rng.integers(0, 24)), minutes=int(rng.integers(0, 60)))
                ct = 40 * rng.uniform(0.95, 1.08)
                if ln == 'L3' and day < F['L3-M1']:
                    f = min(1, max(0, (day - D('2024-12-01')).days / (F['L3-M1'] - D('2024-12-01')).days))
                    ct = 40 * (1.03 + 0.5 * f ** 1.3) * rng.uniform(0.97, 1.03)
                add(t, f'{ln}-M1', 'Trip', round(ct, 1)); add(t, f'{ln}-M2', 'Trip')
        for tr in ['T1', 'T2']:
            if rng.random() < 1 / 200:
                t = day + pd.Timedelta(hours=10); add(t, f'{tr}-87', 'Trip', round(45 * rng.uniform(0.95, 1.08), 1))
        # scenarios
        if D('2025-05-01') <= day < F['T1-87']:
            f = (day - D('2025-05-01')).days / (F['T1-87'] - D('2025-05-01')).days
            if rng.random() < 0.01 + 0.10 * f ** 1.5:
                add(day + pd.Timedelta(hours=3), 'T1-87', 'Self-test alarm')
            if day >= D('2025-07-15') and rng.random() < 0.03 + 0.05 * f:
                add(day + pd.Timedelta(hours=4), 'T1-87', 'Reboot')
            if day >= D('2025-08-20') and rng.random() < 0.04:
                add(day + pd.Timedelta(hours=5), 'T1-87', 'Power supply alarm')
        if day == F['T1-87'] - pd.Timedelta(days=1):
            add(day + pd.Timedelta(hours=22), 'T1-87', 'Relay failure')
        if D('2025-12-01') <= day < F['L4-M2']:
            f = (day - D('2025-12-01')).days / (F['L4-M2'] - D('2025-12-01')).days
            if rng.random() < 0.015 + 0.30 * f ** 2:
                add(day + pd.Timedelta(hours=int(rng.integers(0, 24))), 'L4-M2', 'Comms failure')
        if D('2026-02-01') <= day < F['BB-87A']:
            f = (day - D('2026-02-01')).days / (F['BB-87A'] - D('2026-02-01')).days
            if rng.random() < 0.02 + 0.20 * f ** 2:
                add(day + pd.Timedelta(hours=int(rng.integers(0, 24))), 'BB-87A', 'CT supervision')
        if day == F['L2-M1']:
            add(day + pd.Timedelta(hours=14, minutes=7), 'L2-M1', 'Mal-operation')
        for rid, fd in F.items():
            if day == fd + pd.Timedelta(days=1 if rid == 'L2-M1' else 0):
                add(day + pd.Timedelta(hours=23), rid, 'Maintenance')
    events = pd.DataFrame(ev, columns=['Timestamp', 'Relay ID', 'Event', 'Value']).sort_values('Timestamp').reset_index(drop=True)

    # ---- disturbance records: one per trip plus pickups without trip; features drawn from the shape of the real fleet
    REC_MEAN, REC_COV = [2.2146, 0.0127], [[2.2469, 1.4355], [1.4355, 2.2231]]
    rec = []
    trips = events[events['Event'] == 'Trip'][['Timestamp', 'Relay ID']].values.tolist()
    extra = [[day + pd.Timedelta(hours=int(rng.integers(0, 24))), rid] for day in days for rid in reg['Relay ID']
             if rng.random() < (1 / 8 if (rid == 'T1-87' and D('2025-04-01') <= day < F['T1-87']) else 1 / 25)]
    for t, rid in trips + extra:
        t = pd.Timestamp(t)
        lc, ln = rng.multivariate_normal(REC_MEAN, np.array(REC_COV) * 0.3)      # healthy relays: typical records
        if rid == 'T1-87' and D('2025-04-01') <= t < F['T1-87']:
            f = (t - D('2025-04-01')).days / (F['T1-87'] - D('2025-04-01')).days
            lc, ln = rng.normal(2.0, 0.2), rng.normal(0.3, 0.25) + 3.4 * f ** 0.8    # analog input noise growing on a failing card
        rec.append([t, rid, round(10 ** lc, 2), round(10 ** ln, 4)])
    records = pd.DataFrame(rec, columns=['Timestamp', 'Relay ID', 'Max Fault Current (A)', 'Noise Index']).sort_values('Timestamp').reset_index(drop=True)
    records['Timestamp'] = pd.to_datetime(records['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    events['Timestamp'] = pd.to_datetime(events['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    findings_ = pd.DataFrame([
        ['T1-87', '2025-10-05', 'Relay failure: power supply card failed and replaced'],
        ['L3-M1', '2025-11-20', 'Breaker Q3-CB1 overhaul: worn trip latch, dry mechanism'],
        ['L2-M1', '2026-03-10', 'Slow zone-2 trip during an external fault; timer card replaced'],
        ['BB-87A', '2026-05-20', 'Loose CT secondary terminal found and tightened'],
        ['L4-M2', '2026-06-15', 'Protection signalling fibre channel repaired']], columns=['Relay ID', 'Date', 'Event'])
    return {'register': reg, 'tests': tests, 'events': events, 'records': records, 'findings': findings_}
