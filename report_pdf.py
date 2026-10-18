"""RelayGuard AI - PDF reports (disturbance record, OMICRON test, combined report)."""
import io
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

NAVY = colors.HexColor('#0b2e59'); LINE = colors.HexColor('#d5dce6'); MUTED = colors.HexColor('#5a6575')
SCOL = {'EXCELLENT': colors.HexColor('#1a9e5c'), 'WATCH': colors.HexColor('#d99a00'), 'ACTION': colors.HexColor('#d63a3a')}
SLAB = {'EXCELLENT': 'NORMAL', 'WATCH': 'WATCH', 'ACTION': 'ACTION REQUIRED'}
W = A4[0] - 30 * mm

_ss = getSampleStyleSheet()
H1 = ParagraphStyle('h1', parent=_ss['Heading1'], fontName='Helvetica-Bold', fontSize=17, textColor=NAVY, spaceAfter=4)
H2 = ParagraphStyle('h2', parent=_ss['Heading2'], fontName='Helvetica-Bold', fontSize=12, textColor=NAVY, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle('b', parent=_ss['BodyText'], fontName='Helvetica', fontSize=9.2, leading=12.5)
SMALL = ParagraphStyle('s', parent=BODY, fontSize=7.8, leading=10, textColor=MUTED)
CELL = ParagraphStyle('c', parent=BODY, fontSize=8, leading=10)
CELLB = ParagraphStyle('cb', parent=CELL, fontName='Helvetica-Bold', textColor=colors.white)

_MAP = {'Ω': 'ohm', '≥': '>=', '≤': '<=', '−': '-', '…': '...', '→': '->', '×': 'x', '÷': '/', '²': '2', '·': '-', '’': "'", '‘': "'",
        '“': '"', '”': '"', '–': '-', '—': '-', '±': '+/-', '°': ' deg'}


def clean(x):
    s = '' if x is None else str(x)
    for k, v in _MAP.items():
        s = s.replace(k, v)
    s = s.encode('latin-1', 'replace').decode('latin-1')
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def P(x, st=BODY):
    return Paragraph(clean(x), st)


def table(rows, widths, head=True, zebra=True, align_right=()):
    data = [[Paragraph(clean(c), CELLB if (head and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [('GRID', (0, 0), (-1, -1), 0.4, LINE), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
          ('TOPPADDING', (0, 0), (-1, -1), 2.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5)]
    if head:
        st.append(('BACKGROUND', (0, 0), (-1, 0), NAVY))
    if zebra:
        for i in range(1 if head else 0, len(rows)):
            if i % 2 == 0:
                st.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f5f7fa')))
    t.setStyle(TableStyle(st))
    return t


def banner(text, sub, status):
    c = SCOL.get(status, NAVY)
    t = Table([[Paragraph(f"<font color='{c.hexval()}' size='14'><b>{clean(text)}</b></font>", BODY)], [P(sub)]], colWidths=[W])
    t.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 1.4, c), ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                           ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
    return t


def fig_image(fig, width=W):
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=160, bbox_inches='tight'); plt.close(fig); buf.seek(0)
    from PIL import Image as PI
    w, h = PI.open(buf).size; buf.seek(0)
    return Image(buf, width=width, height=width * h / w)


def _doc(title):
    buf = io.BytesIO()

    def deco(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(NAVY); canvas.rect(0, A4[1] - 12 * mm, A4[0], 12 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white); canvas.setFont('Helvetica-Bold', 10)
        canvas.drawString(15 * mm, A4[1] - 8 * mm, 'RelayGuard AI'); canvas.setFont('Helvetica', 8.5)
        canvas.drawRightString(A4[0] - 15 * mm, A4[1] - 8 * mm, clean(title).replace('&amp;', '&'))
        canvas.setFillColor(MUTED); canvas.setFont('Helvetica', 7.5)
        canvas.drawString(15 * mm, 9 * mm, f"Generated {datetime.now():%Y-%m-%d %H:%M}  -  PowerGuard AI  -  Confidential")
        canvas.drawRightString(A4[0] - 15 * mm, 9 * mm, f"Page {doc.page}")
        canvas.restoreState()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=18 * mm, bottomMargin=16 * mm,
                            title=title, author='RelayGuard AI')
    return buf, doc, deco


def _fmt(v, d=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '-'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:,.{d}f}" if abs(v) >= 1 else f"{v:.4g}"
    return str(v)


# ---------------------------------------------------------------- disturbance record
def record_pdf(name, feat, ai_pct, ai_risk, status, notes, cfg, t, A, grid=None, n_train=0):
    buf, doc, deco = _doc('Disturbance Record Analysis')
    S = [P('Disturbance Record Analysis', H1),
         P(f"Record: {name}"),
         Spacer(1, 4)]
    S.append(banner(f"{SLAB[status]}  -  AI record-anomaly risk {ai_risk:.0f}%",
                    f"AI percentile {ai_pct:.0f}: more unusual than {ai_pct:.0f}% of the {n_train} real disturbance records the model learned from", status))
    S.append(Spacer(1, 6))
    S.append(P('Record', H2))
    meta = [['Item', 'Value', 'Item', 'Value'],
            ['Station (file)', feat.get('Station (cfg)') or '-', 'Device (file)', feat.get('Device (cfg)') or '-'],
            ['Record time', _fmt(feat.get('Record Time')), 'Clock', 'NOT synchronised' if feat.get('Clock Invalid') else 'OK'],
            ['Sample rate', f"{_fmt(feat.get('Sample Rate (Hz)'), 0)} Hz", 'Duration', f"{_fmt(feat.get('Duration (ms)'), 0)} ms"],
            ['Analog / digital channels', f"{feat.get('Analog Channels')} / {feat.get('Digital Channels')}", 'Record type',
             'Likely secondary injection test' if feat.get('Likely Test Record') else 'Fault / disturbance']]
    S.append(table(meta, [W * 0.22, W * 0.28, W * 0.2, W * 0.3]))
    S.append(P('Results', H2))
    res = [['Quantity', 'Value', 'Quantity', 'Value'],
           ['Max fault current (AI input)', f"{_fmt(feat.get('Max Fault Current (A)'))} A", 'Noise index (AI input)', _fmt(feat.get('Noise Index'), 3)],
           ['Fault inception', f"{_fmt(feat.get('Fault Inception (ms)'))} ms", 'Trip in record', f"{_fmt(feat.get('Trip Time In Record (ms)'))} ms"],
           ['Relay operate time', f"{_fmt(feat.get('Relay Operate Time (ms)'))} ms", 'Breaker clearing time', f"{_fmt(feat.get('Breaker Clearing Time (ms)'))} ms"],
           ['Fault current (rms)', f"{_fmt(feat.get('Max Fault Current (rms)'))}", 'Pre-fault current (rms)', f"{_fmt(feat.get('Prefault Current (rms)'))}"],
           ['Pre-fault noise', f"{_fmt(feat.get('Prefault Noise (% of fault)'), 3)} % of fault", 'Pre-fault imbalance', f"{_fmt(feat.get('Prefault Imbalance (%)'))} %"],
           ['CT residual mismatch', f"{_fmt(feat.get('CT Residual Mismatch (% of fault)'), 2)} % of fault", 'Voltage dip (min / pre-fault)', _fmt(feat.get('Min Voltage / Prefault'), 3)]]
    S.append(table(res, [W * 0.25, W * 0.25, W * 0.25, W * 0.25]))
    if notes:
        S.append(P('Findings', H2))
        for n in notes:
            S.append(P(f"- {n}"))
    # waveform
    I = [i for i, c in enumerate(cfg['analog']) if c['unit'].upper() in ('A', 'KA')][:4]
    V = [i for i, c in enumerate(cfg['analog']) if c['unit'].upper() in ('V', 'KV')][:3]
    ms = (t - t[0]) * 1000
    fig, axs = plt.subplots(2 if V else 1, 1, figsize=(10, 5.2 if V else 2.8), sharex=True, squeeze=False)
    for i in I:
        axs[0, 0].plot(ms, A[:, i], lw=0.6, label=cfg['analog'][i]['name'])
    axs[0, 0].set_ylabel('Current'); axs[0, 0].legend(fontsize=7, ncol=4, loc='upper right'); axs[0, 0].grid(alpha=.25)
    if V:
        for i in V:
            axs[1, 0].plot(ms, A[:, i], lw=0.6, label=cfg['analog'][i]['name'])
        axs[1, 0].set_ylabel('Voltage'); axs[1, 0].legend(fontsize=7, ncol=3, loc='upper right'); axs[1, 0].grid(alpha=.25)
    for key, col, lab in [('Fault Inception (ms)', '#d99a00', 'Fault'), ('Trip Time In Record (ms)', '#d63a3a', 'Trip')]:
        v = feat.get(key)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            for ax in axs[:, 0]:
                ax.axvline(float(v), color=col, ls='--', lw=1)
            axs[0, 0].text(float(v), axs[0, 0].get_ylim()[1], f' {lab}', color=col, fontsize=8, va='top')
    axs[-1, 0].set_xlabel('ms from record start')
    S.append(KeepTogether([P('Waveforms', H2), fig_image(fig)]))
    if grid is not None:
        lc, ln, Z = grid
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        cs = ax.contourf(lc, ln, Z, levels=[0, 50, 80, 90, 95, 99, 100.01], colors=['#e8f5ee', '#d4ecdc', '#fbe8b3', '#f6c98a', '#f2a6a6', '#d63a3a'])
        fig.colorbar(cs, ax=ax, label='AI percentile')
        ax.plot(np.log10(max(feat.get('Max Fault Current (A)') or 0.01, 0.01)), np.log10(max(feat.get('Noise Index') or 1e-4, 1e-4)), 'o',
                color='#0b2e59', ms=9, mec='white')
        ax.set_xlabel('Max fault current, log10 (A)'); ax.set_ylabel('Noise index, log10')
        ax.set_title(f'This record against what the AI learned from {n_train} real records', fontsize=9, color='#0b2e59')
        S.append(KeepTogether([P('AI model', H2), fig_image(fig, W * 0.8),
                               P('The AI model (Isolation Forest) learned the normal behaviour of the fleet from real disturbance records. '
                                 'The percentile says how unusual this record is. A relay\'s AI risk uses the median of its last 3 records: '
                                 'it starts at percentile 90 and reaches 90% at percentile 99.5.', SMALL)]))
    doc.build(S, onFirstPage=deco, onLaterPages=deco)
    return buf.getvalue()


# ---------------------------------------------------------------- OMICRON test
def test_pdf(rid, d, pts, reg, reasons):
    buf, doc, deco = _doc('Relay Test Report')
    n_f = sum(1 for p in pts if not p['Pass'])
    S = [P('Relay Test Report', H1), Spacer(1, 3)]
    S.append(banner('ALL TEST POINTS PASS' if not n_f else f'{n_f} TEST POINT(S) OUTSIDE TOLERANCE', f"{rid}  -  test of {d}",
                    'EXCELLENT' if not n_f else 'ACTION'))
    S.append(Spacer(1, 6))
    S.append(table([['Relay', rid, 'Bay', reg.get('Bay', '') or '-'], ['Functions', reg.get('Functions', '') or '-', 'Technology', reg.get('Technology', '') or '-'],
                    ['Test date', str(d), 'Test points', str(len(pts))]], [W * 0.17, W * 0.33, W * 0.17, W * 0.33], head=False))
    S.append(P('Test points', H2))
    rows = [['Function', 'Test point', 'Quantity', 'Unit', 'Nominal', 'Measured', 'Tolerance', 'Error % tol', 'Drift % tol', 'Result']]
    for p in pts:
        rows.append([p.get('Function'), p.get('Test Point'), p.get('Quantity'), p.get('Unit') or '', _fmt(float(p['Nominal']), 3), _fmt(float(p['Measured']), 3),
                     '+/- ' + _fmt(float(p['Tolerance']), 3), f"{p['Error (% of tol)']:.0f}", '' if p['Drift (% of tol)'] is None else f"{p['Drift (% of tol)']:+.0f}",
                     'PASS' if p['Pass'] else 'FAIL'])
    t = table(rows, [W * x for x in [0.1, 0.14, 0.1, 0.06, 0.1, 0.1, 0.11, 0.09, 0.09, 0.08]])
    for i, p in enumerate(pts, 1):
        t.setStyle(TableStyle([('TEXTCOLOR', (9, i), (9, i), SCOL['EXCELLENT'] if p['Pass'] else SCOL['ACTION'])]))
    S.append(t)
    if pts:
        fig, ax = plt.subplots(figsize=(9, 0.45 * len(pts) + 1.2))
        names = [' '.join(str(p.get(k, '')) for k in ['Function', 'Test Point', 'Quantity']) for p in pts][::-1]
        err = [p['Error (% of tol)'] for p in pts][::-1]
        ax.barh(names, err, color=['#1a9e5c' if e <= 50 else '#d99a00' if e <= 100 else '#d63a3a' for e in err])
        ax.axvline(100, color='#d63a3a', ls='--', lw=1); ax.set_xlabel('% of tolerance'); ax.grid(axis='x', alpha=.25)
        S.append(KeepTogether([P('Error as % of the tolerance', H2), fig_image(fig, W * 0.9)]))
    S.append(P('Assessment', H2))
    for r in reasons:
        S.append(P(f"- {r}"))
    S.append(Spacer(1, 6)); S.append(P('Error = |measured - nominal| / tolerance. Drift = change of the error since the previous test of the same point.', SMALL))
    doc.build(S, onFirstPage=deco, onLaterPages=deco)
    return buf.getvalue()


# ---------------------------------------------------------------- combined report
def combined_pdf(R, summary, source, n_a, n_w):
    buf, doc, deco = _doc('Combined Protection Condition Report')
    S = [P('Combined Protection Condition Report', H1), P(f"{len(R)} relays  -  source: {source}", SMALL), Spacer(1, 4)]
    st = 'ACTION' if n_a else 'WATCH' if n_w else 'EXCELLENT'
    S.append(banner(f"{n_a} action  -  {n_w} watch  -  {len(R) - n_a - n_w} normal", summary, st))
    S.append(P('All relays', H2))
    rows = [['Relay', 'Bay', 'Role', 'Governing mode', 'Risk %', 'Status', 'RPN', 'Level']]
    for _, r in R.iterrows():
        rows.append([r['Relay ID'], r['Bay'], r['Role'], r['Governing Mode'], f"{r['Overall Risk (%)']:.0f}", r['Status'], str(r['RPN']), r['RPN Level']])
    t = table(rows, [W * x for x in [0.11, 0.15, 0.1, 0.26, 0.08, 0.12, 0.07, 0.11]])
    for i, (_, r) in enumerate(R.iterrows(), 1):
        t.setStyle(TableStyle([('TEXTCOLOR', (5, i), (5, i), SCOL[r['Status']])]))
    S.append(t)
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(R) + 1))
    rr = R.sort_values('Overall Risk (%)')
    ax.barh(rr['Relay ID'], rr['Overall Risk (%)'], color=[{'EXCELLENT': '#1a9e5c', 'WATCH': '#d99a00', 'ACTION': '#d63a3a'}[s] for s in rr['Status']])
    for x in (35, 70):
        ax.axvline(x, color='#888', ls='--', lw=1)
    ax.set_xlim(0, 105); ax.set_xlabel('Failure risk (%)'); ax.grid(axis='x', alpha=.25)
    S.append(KeepTogether([P('Risk ranking', H2), fig_image(fig, W * 0.9)]))
    above = R[R['Status'] != 'EXCELLENT']
    if len(above):
        S.append(P('Relays above the normal band', H2))
        for _, r in above.iterrows():
            block = [Paragraph(f"<b>{clean(r['Relay ID'])}</b>  -  {clean(r['Governing Mode'])}  -  {r['Overall Risk (%)']:.0f}% {r['Status']}", BODY)]
            block += [P(f"- {f}") for f in r['Findings']]
            block += [P(f"RPN = {r['S']} x {r['O']} x {r['D']} = {r['RPN']} ({r['RPN Level']}).  Action: {r['Action']}", SMALL), Spacer(1, 5)]
            S.append(KeepTogether(block))
    doc.build(S, onFirstPage=deco, onLaterPages=deco)
    return buf.getvalue()
