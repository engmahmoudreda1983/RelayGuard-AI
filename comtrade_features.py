"""RelayGuard AI - COMTRADE (IEEE C37.111) reader and disturbance-record features, used by the app."""
import os, re, sys, csv, traceback
from datetime import datetime
import numpy as np

# ---------------------------------------------------------------- COMTRADE reader (IEEE C37.111, ASCII and BINARY)
def _num(s, d=0.0):
    try:
        return float(str(s).strip())
    except Exception:
        return d


def read_cfg(path=None, text=None):
    L = [l.rstrip('\r\n') for l in (text.splitlines() if text is not None else open(path, encoding='latin-1'))]
    head = L[0].split(',')
    c = L[1].split(',')
    nA = int(re.sub(r'\D', '', c[1]) or 0); nD = int(re.sub(r'\D', '', c[2]) or 0)
    ana, dig = [], []
    for i in range(nA):
        f = L[2 + i].split(',')
        ana.append({'name': f[1].strip(), 'unit': (f[4].strip() if len(f) > 4 else ''), 'a': _num(f[5] if len(f) > 5 else 1, 1),
                    'b': _num(f[6] if len(f) > 6 else 0)})
    for i in range(nD):
        f = L[2 + nA + i].split(',')
        dig.append({'name': f[1].strip() if len(f) > 1 else ''})
    k = 2 + nA + nD
    freq = _num(L[k], 50) or 50; k += 1
    nrates = int(_num(L[k])); k += 1
    rates = []
    for _ in range(max(nrates, 1)):
        f = L[k].split(','); rates.append((_num(f[0]), int(_num(f[1] if len(f) > 1 else 0)))); k += 1

    def ts(s):
        s = s.strip()
        for fmt in ('%d/%m/%Y,%H:%M:%S.%f', '%m/%d/%Y,%H:%M:%S.%f', '%d/%m/%y,%H:%M:%S.%f', '%d/%m/%Y,%H:%M:%S'):
            try:
                return datetime.strptime(s[:26], fmt)
            except Exception:
                pass
        return None
    t_first, t_trig = ts(L[k]), ts(L[k + 1]); k += 2
    ft = L[k].strip().upper() if k < len(L) else 'ASCII'; k += 1
    tmult = _num(L[k], 1) if k < len(L) and L[k].strip() else 1.0
    return {'station': head[0].strip(), 'device': head[1].strip() if len(head) > 1 else '', 'analog': ana, 'digital': dig,
            'freq': freq, 'rates': rates, 't_first': t_first, 't_trigger': t_trig, 'format': ft, 'timemult': tmult or 1.0}


def read_dat(path, cfg, data=None):
    nA, nD = len(cfg['analog']), len(cfg['digital'])
    if cfg['format'].startswith('ASCII'):
        good = []
        for line in (data.decode('latin-1').splitlines() if data is not None else open(path, encoding='latin-1')):
            f = line.strip().split(',')
            if len(f) >= 2 + nA + nD:
                try:
                    good.append([float(x) if x.strip() not in ('', '99999') else np.nan for x in f[:2 + nA + nD]])
                except ValueError:
                    pass
        rows = np.array(good, dtype=float).reshape(-1, 2 + nA + nD)
        t = rows[:, 1] * cfg['timemult']; A = rows[:, 2:2 + nA]; D = rows[:, 2 + nA:].astype(int)
    else:
        size = 4 if 'FLOAT32' in cfg['format'] or 'BINARY32' in cfg['format'] else 2
        nw = (nD + 15) // 16
        atype = '<f4' if 'FLOAT32' in cfg['format'] else ('<i4' if size == 4 else '<i2')
        dt = np.dtype([('n', '<u4'), ('t', '<u4'), ('a', atype, (nA,)), ('d', '<u2', (nw,))])
        raw = data if data is not None else open(path, 'rb').read(); N = len(raw) // dt.itemsize
        arr = np.frombuffer(raw[:N * dt.itemsize], dtype=dt)
        t = arr['t'].astype(float) * cfg['timemult']; A = arr['a'].astype(float)
        D = np.unpackbits(arr['d'].view(np.uint8).reshape(N, -1), axis=1, bitorder='little')[:, :nD].astype(int) if nD else np.zeros((N, 0), int)
    a = np.array([c['a'] for c in cfg['analog']]); b = np.array([c['b'] for c in cfg['analog']])
    A = A * a + b
    sr = cfg['rates'][0][0] if cfg['rates'] else 0
    if len(t) < 2 or not np.all(np.diff(t) > 0):
        t = np.arange(len(A)) / (sr or 1000.0) * 1e6
    return t / 1e6, A, D


# ---------------------------------------------------------------- features
def cycle_rms(x, w):
    n = len(x) // w
    if n == 0:
        return np.array([])
    return np.sqrt(np.nanmean(x[:n * w].reshape(n, w) ** 2, axis=1))


def fund_residual(x, t, f0):
    """RMS of what is left after removing the fundamental (least-squares fit): noise and distortion."""
    if len(x) < 8 or np.all(np.isnan(x)):
        return np.nan, np.nan
    M = np.c_[np.sin(2 * np.pi * f0 * t), np.cos(2 * np.pi * f0 * t), np.ones_like(t)]
    ok = ~np.isnan(x)
    coef, *_ = np.linalg.lstsq(M[ok], x[ok], rcond=None)
    res = x[ok] - M[ok] @ coef
    return float(np.sqrt(np.mean(res ** 2))), float(np.hypot(coef[0], coef[1]) / np.sqrt(2))


def pick(chs, pats, unit=None):
    out = []
    for i, c in enumerate(chs):
        n = c['name'].lower().replace(' ', '')
        if unit and c['unit'].upper().strip() not in unit:
            continue
        if any(re.search(p, n) for p in pats):
            out.append(i)
    return out


def features(cfg_path=None, dat_path=None, root=None, cfg_text=None, dat_bytes=None, name=None):
    cfg = read_cfg(cfg_path, cfg_text); t, A, D = read_dat(dat_path, cfg, dat_bytes)
    f0 = cfg['freq'] or 50.0
    dt_s = float(np.median(np.diff(t))) if len(t) > 1 else 0.001
    sr = 1.0 / dt_s; w = max(1, int(round(sr / f0)))
    rel = name or (os.path.relpath(cfg_path, root) if root else os.path.basename(cfg_path or 'record.cfg'))
    parts = rel.replace('\\', '/').split('/')
    r = {'Record Path': rel, 'Folder 1': parts[0] if len(parts) > 1 else '', 'Folder 2': parts[1] if len(parts) > 2 else '',
         'Folder 3': parts[2] if len(parts) > 3 else '', 'Station (cfg)': cfg['station'], 'Device (cfg)': cfg['device'],
         'Record Time': cfg['t_trigger'] or cfg['t_first'], 'Sample Rate (Hz)': round(sr, 1), 'Duration (ms)': round((t[-1] - t[0]) * 1000, 1) if len(t) else 0,
         'Analog Channels': len(cfg['analog']), 'Digital Channels': len(cfg['digital'])}
    # same definitions as the fleet data the AI model was trained on: first analog channel
    if A.shape[1]:
        x0 = A[:, 0]
        r['Max Fault Current (A)'] = round(float(np.nanmax(np.abs(x0))), 4)
        r['Noise Index'] = round(float(np.nanstd(x0[:10], ddof=1)), 4) if len(x0) >= 3 else None
    yr = r['Record Time'].year if r['Record Time'] else None
    r['Clock Invalid'] = int(yr is None or yr < 2000 or yr > datetime.now().year)
    I = pick(cfg['analog'], [r'^i?[abc123lr]$', r'^i[abc]', r'current.*i[abc]', r'^i(l)?[123]', r'phase[abc]'], {'A', 'KA'})
    I = [i for i in I if not re.search(r'n|y|sens|m$|0', cfg['analog'][i]['name'].lower().replace('current', ''))] or \
        [i for i, c in enumerate(cfg['analog']) if c['unit'].upper() in ('A', 'KA')][:3]
    IN = [i for i, c in enumerate(cfg['analog']) if c['unit'].upper() in ('A', 'KA') and re.search(r'\bin\b|^in$|^in |iy|ie|i0|ires|neutral', c['name'].lower()) and 'sens' not in c['name'].lower()]
    V = [i for i, c in enumerate(cfg['analog']) if c['unit'].upper() in ('V', 'KV') and not re.search(r'sync|check|vn|v0|ve', c['name'].lower())][:3]
    r['Current Channels'] = len(I); r['Voltage Channels'] = len(V)
    if not I or len(t) < 3 * w:
        r['Note'] = 'no phase currents found'; return r
    rms = np.vstack([cycle_rms(A[:, i], w) for i in I])          # phase x cycle
    imax = np.nanmax(rms, axis=0)
    pre_n = max(1, min(int(len(imax) * 0.15), 5))
    pre = float(np.nanmedian(imax[:pre_n]))
    peak = float(np.nanmax(imax))
    r['Prefault Current (rms)'] = round(pre, 3); r['Max Fault Current (rms)'] = round(peak, 3)
    r['Fault/Prefault Ratio'] = round(peak / pre, 2) if pre > 0 else None
    thr = max(2.0 * pre, pre + 0.2 * (peak - pre))
    inc = int(np.argmax(imax > thr)) if peak > thr else None
    r['Fault Inception (ms)'] = round(inc * w * dt_s * 1000, 1) if inc is not None else None
    # pre-fault noise: residual after removing the fundamental, in % of the fault current (robust when the line is unloaded)
    end = (inc if inc else len(imax)) * w
    nz = []
    for i in I:
        res, fund = fund_residual(A[:max(end - w, 2 * w), i], t[:max(end - w, 2 * w)], f0)
        nz.append(res)
    r['Prefault Noise (rms)'] = round(float(np.nanmean(nz)), 4)
    r['Prefault Noise (% of fault)'] = round(100 * float(np.nanmean(nz)) / peak, 3) if peak > 0 else None
    r['Prefault Imbalance (%)'] = round(100 * float((np.nanmax(rms[:, :pre_n].mean(1)) - np.nanmin(rms[:, :pre_n].mean(1))) / max(pre, 1e-9)), 1) if pre > 0 else None
    # CT check: measured neutral against the sum of the phases
    if IN and len(I) >= 3:
        s3 = A[:, I[0]] + A[:, I[1]] + A[:, I[2]]
        mis = cycle_rms(A[:, IN[0]] - s3, w)
        r['CT Residual Mismatch (% of fault)'] = round(100 * float(np.nanmax(mis)) / peak, 2) if peak > 0 else None
    # trips and breaker
    names = [d['name'].lower() for d in cfg['digital']]
    trip_idx = [j for j, n in enumerate(names) if re.search(r'trip', n) and not re.search(r'block|ext|reset|latch|test|sig\.ref', n)]
    t_trip = None
    for j in trip_idx:
        ch = D[:, j]
        if (ch != ch[0]).any() and ch[0] == 0:
            k = int(np.argmax(ch != 0)); tt = t[k] - t[0]
            t_trip = tt if t_trip is None else min(t_trip, tt)
    r['Trip Time In Record (ms)'] = round(t_trip * 1000, 1) if t_trip is not None else None
    r['Relay Operate Time (ms)'] = round(t_trip * 1000 - r['Fault Inception (ms)'], 1) if (t_trip is not None and inc is not None) else None
    if t_trip is not None and peak > 0:
        kt = int(t_trip / (w * dt_s))
        after = imax[kt:]
        off = np.nonzero(after < max(0.1 * peak, 2 * pre if pre < 0.1 * peak else 0.1 * peak))[0]
        r['Current Interrupted'] = int(len(off) > 0)
        r['Breaker Clearing Time (ms)'] = round(float(off[0]) * w * dt_s * 1000, 1) if len(off) else None
    cbo = [j for j, n in enumerate(names) if re.search(r'cb.*open|52b|cb.*off', n) and 'ext' not in n]
    if t_trip is not None:
        for j in cbo:
            ch = D[:, j]
            if (ch != ch[0]).any():
                k = int(np.argmax(ch != ch[0])); r['CB Aux Open After Trip (ms)'] = round((t[k] - t[0] - t_trip) * 1000, 1); break
    if V:
        vr = np.vstack([cycle_rms(A[:, i], w) for i in V]); vmin = np.nanmin(vr, axis=0)
        vpre = float(np.nanmedian(vmin[:pre_n]))
        r['Prefault Voltage (rms)'] = round(vpre, 2)
        r['Min Voltage / Prefault'] = round(float(np.nanmin(vmin)) / vpre, 3) if vpre > 0 else None
        n0 = max(2 * w, (inc or len(imax)) * w - w)
        res, fund = fund_residual(A[:n0, V[0]], t[:n0], f0)
        r['Voltage Present'] = int(fund > 3 * res) if not np.isnan(res) else None
    # relay status flags inside the record
    st = lambda pat: int(any(D[0, j] == 1 for j, n in enumerate(names) if re.search(pat, n)))
    r['Flag Not Ready'] = st(r'not ready|faulty|relay fail|watchdog')
    r['Flag Test Mode'] = st(r'test mode')
    r['Flag Outputs Blocked'] = st(r'outp.*block|trip.*block')
    # likely a secondary injection test rather than a real fault: tripped, current never interrupted, no voltage dip
    r['Likely Test Record'] = int(bool(r.get('Flag Test Mode')) or (t_trip is not None and r.get('Current Interrupted') == 0 and
                                    (not r.get('Voltage Present') or (r.get('Min Voltage / Prefault') or 1) > 0.9)))
    return r




def load(cfg_text, dat_bytes):
    """Parse one record from memory: returns cfg, time (s), analog matrix, digital matrix."""
    cfg = read_cfg(text=cfg_text)
    t, A, D = read_dat(None, cfg, dat_bytes)
    return cfg, t, A, D


def demo_record(seed=4):
    """A synthetic phase-to-earth fault record (ASCII COMTRADE), for trying the page without real files."""
    rng = np.random.default_rng(seed)
    sr, f0, n = 2000, 50.0, 1600
    t = np.arange(n) / sr
    inc, trip, clear = 0.200, 0.262, 0.304
    Ipre, If, Vn = 25.0, 4200.0, 63.5
    ia = np.where(t < inc, Ipre, np.where(t < clear, If, 0)) * np.sqrt(2) * np.sin(2 * np.pi * f0 * t - 0.4)
    ib = np.where(t < clear, Ipre, 0) * np.sqrt(2) * np.sin(2 * np.pi * f0 * t - 0.4 - 2.094)
    ic = np.where(t < clear, Ipre, 0) * np.sqrt(2) * np.sin(2 * np.pi * f0 * t - 0.4 + 2.094)
    va = np.where((t >= inc) & (t < clear), 0.35, 1.0) * Vn * np.sqrt(2) * np.sin(2 * np.pi * f0 * t)
    vb = Vn * np.sqrt(2) * np.sin(2 * np.pi * f0 * t - 2.094); vc = Vn * np.sqrt(2) * np.sin(2 * np.pi * f0 * t + 2.094)
    I = np.c_[ia, ib, ic] + rng.normal(0, 3.0, (n, 3)); IN = I.sum(1) + rng.normal(0, 1.0, n)
    V = np.c_[va, vb, vc] + rng.normal(0, 0.3, (n, 3))
    trip_d = (t >= trip).astype(int); cbo = (t >= clear - 0.004).astype(int)
    a = [0.2, 0.2, 0.2, 0.2, 0.01, 0.01, 0.01]
    X = np.c_[I, IN, V]
    q = np.round(X / np.array(a)).astype(int)
    cfg = ["DEMO SUBSTATION,LINE-9 MAIN 1,1999", "9,7A,2D"]
    for k, (nm, u) in enumerate([('IA', 'A'), ('IB', 'A'), ('IC', 'A'), ('IN', 'A'), ('VA', 'kV'), ('VB', 'kV'), ('VC', 'kV')]):
        cfg.append(f"{k + 1},{nm},,,{'A' if u == 'A' else 'V'},{a[k]},0,0,-32767,32767,2000,1,S")
    cfg += ["1,3 PH TRIP,,,0", "2,CB OPEN,,,0", "50", "1", f"{sr},{n}", "01/03/2026,10:15:22.800000", "01/03/2026,10:15:23.000000", "ASCII", "1"]
    rows = [f"{i + 1},{int(round(t[i] * 1e6))}," + ",".join(str(v) for v in q[i]) + f",{trip_d[i]},{cbo[i]}" for i in range(n)]
    return "\n".join(cfg), ("\n".join(rows)).encode()
