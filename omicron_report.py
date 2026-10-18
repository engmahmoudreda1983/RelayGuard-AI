"""RelayGuard AI - reader for OMICRON Test Universe reports (PDF or TXT export).

Reads the 'OMICRON AUTOMATIC TESTS' part (Assessment Results / Time Assessment tables of the Ramping,
State Sequencer and similar modules) and, when present, the summary test sheet in front of it.
Returns the relay data and one row per test point: Function, Test Point, Quantity, Unit, Nominal,
Measured, Tolerance, Deviation, Result."""
import io
import re
from datetime import datetime

import pandas as pd

UNIT = {'V': ('V', 1), 'mV': ('V', 1e-3), 'kV': ('V', 1e3), 'KV': ('V', 1e3), 'A': ('A', 1), 'mA': ('A', 1e-3), 'kA': ('A', 1e3),
        's': ('s', 1), 'ms': ('s', 1e-3), 'Ω': ('Ω', 1), 'Ohm': ('Ω', 1), 'ohm': ('Ω', 1), 'Hz': ('Hz', 1), 'mHz': ('Hz', 1e-3),
        '°': ('°', 1), '%': ('%', 1), 'In': ('In', 1), 'Iref': ('Iref', 1)}
VAL = re.compile(r'(?<![\w.])(-?\d+(?:\.\d+)?)\s?(mV|kV|KV|V|mA|kA|A|ms|s|Ω|Ohm|ohm|mHz|Hz|°|%|In|Iref)(?![\w])')
COLS = {'nom': ['Nom.', 'Nom', 'Tnom', 'tnom', 'Nominal'], 'act': ['Act.', 'Act', 'Tact', 'tact', 'Actual'],
        'tolm': ['Tol.-', 'Tol-', 'Tdev-', 'Dev-'], 'tolp': ['Tol.+', 'Tol+', 'Tdev+', 'Dev+'], 'dev': ['Dev.', 'Dev', 'Tdev', 'Deviation'],
        'assess': ['Assess', 'Assessment', 'Result']}
KEYS = {k for v in COLS.values() for k in v}


def text_of(data, name=''):
    """Text of the report with its layout kept (PDF via pypdf layout mode, or a TXT export)."""
    if name.lower().endswith('.pdf') or data[:4] == b'%PDF':
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(data))
        pages = []
        for p in r.pages:
            try:
                pages.append(p.extract_text(extraction_mode='layout'))
            except Exception:
                pages.append(p.extract_text() or '')
        return '\n'.join(pages)
    for enc in ('utf-8-sig', 'utf-16', 'cp1252', 'latin-1'):
        try:
            return data.decode(enc)
        except Exception:
            pass
    return data.decode('latin-1', 'replace')


def _norm_header(line):
    # 'Tdev      -' -> 'Tdev-' (same length so the column positions do not move)
    return re.sub(r'(Tdev|Tol\.?|Dev)(\s+)([-+])', lambda m: (m.group(1) + m.group(3)).ljust(len(m.group(0))), line)


def _columns(line):
    cols = []
    for m in re.finditer(r'\S+', line):
        w = m.group()
        for key, names in COLS.items():
            if w in names:
                cols.append((key, w, (m.start() + m.end()) / 2))
    return cols


def _is_header(line):
    words = set(re.findall(r'\S+', _norm_header(line)))
    return len(words & KEYS) >= 3 and (words & set(COLS['nom'])) and (words & set(COLS['act']))


def _to_base(v, u):
    base, f = UNIT.get(u, (u, 1))
    return float(v) * f, base


def _first(pat, txt, flags=0):
    m = re.search(pat, txt, flags)
    return m.group(1).strip() if m else ''


def relay_data(txt):
    d = {}
    d['Substation'] = _first(r'Substation:\s+(\S[^\n]*?)(?:\s{2,}|\n)', txt) or _first(r'Substation\s{2,}(\S+)', txt)
    d['Bay'] = _first(r'\n\s*Bay:\s+(\S[^\n]*?)(?:\s{2,}|\n)', txt) or _first(r'DIA/Bay\s+(\S+)', txt)
    d['Device Type'] = _first(r'Device type:\s+(\S[^\n]*?)(?:\s{2,}|\n)', txt)
    d['Model'] = _first(r'Type/Model\s+(\S+)', txt) or _first(r'Additional info 2:\s+(\S[^\n]*?)(?:\s{2,}|\n)', txt)
    d['Serial'] = _first(r'Serial/model number:\s+(\S+)', txt) or _first(r'Serial No\.\s+(\S+)', txt)
    d['Description'] = _first(r'Additional info 1:\s+(\S[^\n]*?)(?:\s{2,}|\n)', txt)
    d['Firmware'] = _first(r'Firmware Version\s+(\S+)', txt)
    d['Test Set'] = _first(r'\n\s*(CMC\s?\S+)\s+\S+', txt)
    d['Tested By'] = _first(r'Tested by:\s*([^\n-]+?)\s+-', txt)
    dt = None
    m = re.search(r'Test Start:\s+(\d{1,2}-[A-Za-z]{3}-\d{4})', txt)
    if m:
        dt = datetime.strptime(m.group(1), '%d-%b-%Y')
    if dt is None:
        m = re.search(r'Tested by:[^\n]*?(\d{1,2}/\d{1,2}/\d{4})', txt)
        if m:
            dt = datetime.strptime(m.group(1), '%d/%m/%Y')
    if dt is None:
        m = re.search(r'\nDate\s+(\d{1,2}/\d{1,2}/\d{4})', txt)
        if m:
            dt = datetime.strptime(m.group(1), '%m/%d/%Y')
    d['Test Date'] = dt.date() if dt else None
    return d


def _quantity(unit):
    return {'s': 'Trip time', 'Ω': 'Reach', 'Hz': 'Frequency', '°': 'Angle'}.get(unit, 'Pickup')


def _function(group):
    m = re.search(r'\(([^)]+)\)', group or '')
    return m.group(1).strip() if m else (group or '').strip()


def omicron_points(txt):
    lines = txt.splitlines()
    rows, group, module = [], '', ''
    i = 0
    while i < len(lines):
        ln = lines[i]
        g = re.search(r'Group:\s*(.+?)\s*$', ln)
        if g and 'Group end' not in ln:
            group = g.group(1)
        mm = re.search(r'Name:\s+OMICRON(?:\s+(.+?))?(?:\s{2,}|$)', ln)
        if mm:
            module = (mm.group(1) or '').strip()
            if module.startswith('Version') or not module:
                nxt = lines[i + 1].strip() if i + 1 < len(lines) else ''
                module = nxt.split('  ')[0] if nxt and ':' not in nxt.split('  ')[0] else ''
        if _is_header(ln):
            head = _norm_header(ln)
            cols = _columns(head)
            has_act = any(c[1] in ('Act.', 'Act') for c in cols)
            i += 1
            quiet = 0
            while i < len(lines) and not lines[i].strip().startswith('Assess:') and not re.match(r'\s*(Test State|Group)', lines[i]) and quiet < 8:
                row = lines[i]
                quiet = 0 if VAL.search(row) else quiet + 1
                vals = [(m.group(1), m.group(2), (m.start() + m.end()) / 2) for m in VAL.finditer(row)]
                if vals:
                    name = row.strip().split()[0]
                    got = {}
                    for v, u, c in vals:
                        key, w, _ = min(cols, key=lambda col: abs(col[2] - c))
                        if has_act and w in ('Tact', 'tact'):
                            continue                     # pickup time of a ramp, not the assessed value
                        if key in ('nom', 'act', 'tolm', 'tolp', 'dev') and key not in got:
                            got[key] = _to_base(v, u)
                    am = re.search(r'(?<=\s)([+xo])(?=\s|$)', row[int(min(c[2] for c in cols if c[0] in ('dev', 'act', 'nom'))):])
                    if 'nom' in got and 'act' in got:
                        nom, unit = got['nom']; act = got['act'][0]
                        tol = max(abs(got.get('tolm', (0,))[0]), abs(got.get('tolp', (0,))[0])) or None
                        res = {'+': 'Pass', 'x': 'Fail', 'o': 'Not assessed'}.get(am.group(1) if am else '', '')
                        rows.append({'Function': _function(group), 'Group': group, 'Module': module, 'Test Point': name, 'Quantity': _quantity(unit),
                                     'Unit': unit, 'Nominal': nom, 'Measured': act, 'Tolerance': tol,
                                     'Deviation': got.get('dev', (act - nom,))[0], 'Result': res})
                i += 1
            continue
        i += 1
    return rows


def summary_points(txt):
    """Fallback when only the summary test sheet exists: 'Theoretical / Measured / Deviation / Assessment' tables."""
    rows, func, vtol, ttol = [], '', None, None
    for ln in txt.splitlines():
        f = re.search(r'^\s{10,}([A-Za-z][\w /-]+\[\s*[\w/]+\s*\])\s*$', ln)
        if f:
            func = f.group(1).strip()
        t = re.search(r'Tolerance\*?:\s*([\d.]+)\s*%.*?T Tol\.\*?:\s*([\d.]+)\s*s', ln)
        if t:
            vtol, ttol = float(t.group(1)), float(t.group(2))
        m = re.match(r'\s*(\S+)\s+(-?[\d.]+)\s*(\w+|°)\s+(-?[\d.]+)\s*(\w+|°)\s+(-?[\d.]+)\s*(\w+|°)\s+(OK|NOT OK|FAIL|Failed|Passed)\s*$', ln)
        if m and func:
            nom, u = _to_base(m.group(2), m.group(3)); act = _to_base(m.group(4), m.group(5))[0]
            tol = ttol if u == 's' else (vtol / 100 * abs(nom) if vtol else None)
            fn = re.search(r'\[\s*([\w/]+)\s*\]', func)
            rows.append({'Function': fn.group(1) if fn else func, 'Group': func, 'Module': 'Test sheet', 'Test Point': m.group(1),
                         'Quantity': _quantity(u), 'Unit': u, 'Nominal': nom, 'Measured': act, 'Tolerance': tol,
                         'Deviation': act - nom, 'Result': 'Pass' if m.group(8).upper() in ('OK', 'PASSED') else 'Fail'})
    return rows


def parse(data, name=''):
    txt = text_of(data, name)
    info = relay_data(txt)
    pts = omicron_points(txt)
    info['Source'] = 'OMICRON automatic tests' if pts else 'Test sheet'
    if not pts:
        pts = summary_points(txt)
    info['Overall'] = 'Test passed' if pts and all(p['Result'] == 'Pass' for p in pts) else ('Failed points' if any(p['Result'] == 'Fail' for p in pts) else '')
    return info, pd.DataFrame(pts)
