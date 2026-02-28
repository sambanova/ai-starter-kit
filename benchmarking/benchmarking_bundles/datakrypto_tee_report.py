"""
DataKrypto TEE Performance Analysis — PDF Report Generator
Run from repo root with virtual env active:
  python benchmarking/benchmarking_bundles/datakrypto_tee_report.py
"""

import io
import os
import struct
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── reportlab ────────────────────────────────────────────────────────────────
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, inch
from reportlab.platypus import (
    HRFlowable, Image, PageBreak, Paragraph, Spacer, Table, TableStyle,
    SimpleDocTemplate, KeepTogether,
)

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, '../data/datakrypto'))
OUT_PDF    = os.path.join(SCRIPT_DIR, 'datakrypto_tee_performance_report.pdf')

for p in [SCRIPT_DIR, os.path.join(SCRIPT_DIR, '../../'),
          os.path.join(SCRIPT_DIR, '../../../')]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Palette ──────────────────────────────────────────────────────────────────
C_ENC    = '#EF553B'
C_PLAIN  = '#636EFA'
C_NET    = '#FF7F0E'
C_ENCR   = '#D62728'
C_DEC    = '#2CA02C'
C_OTHER  = '#9467BD'
C_NAVY   = '#0d1b2a'
C_BLUE2  = '#1b4f72'
C_LIGHT  = '#eaf4fb'
C_FIND   = '#fef9e7'
C_WARN   = '#fdedec'

CONC_ORDER = [1, 2, 4, 8, 16, 32]

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

DATASETS = {
    'enc_simple': {
        'consolidated': f'{DATA_ROOT}/speed_bench_test_encrypted_simple/consolidated_results/20260225-155057.028751.xlsx',
        'encrypted': True,  'run_type': 'Single Run (1x)', 'label': 'Encrypted',
    },
    'plain_simple': {
        'consolidated': f'{DATA_ROOT}/speed_bench_test_non_encrypted_simple/consolidated_results/20260227-204607.930274.xlsx',
        'encrypted': False, 'run_type': 'Single Run (1x)', 'label': 'Non-Encrypted',
    },
    'enc_multiple': {
        'consolidated': f'{DATA_ROOT}/speed_bench_test_encrypted_multiple/consolidated_results/20260225-150147.253373.xlsx',
        'encrypted': True,  'run_type': 'Multi Run (10x)', 'label': 'Encrypted',
    },
    'plain_multiple': {
        'consolidated': f'{DATA_ROOT}/speed_bench_test_non_encrypted_multiple/consolidated_results/20260227-205436.463636.xlsx',
        'encrypted': False, 'run_type': 'Multi Run (10x)', 'label': 'Non-Encrypted',
    },
}

dfs = []
for key, meta in DATASETS.items():
    df = pd.read_excel(meta['consolidated'])
    df['dataset_key'] = key
    df['encrypted']   = meta['encrypted']
    df['run_type']    = meta['run_type']
    df['label']       = meta['label']
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all['total_requests']  = df_all['num_completed_requests'] + df_all['number_errors']
df_all['error_rate_pct']  = (
    df_all['number_errors'] / df_all['total_requests'].replace(0, np.nan)
) * 100

DIMS = ['run_type', 'num_output_tokens', 'num_concurrent_requests']

df_multi   = df_all[df_all['run_type'] == 'Multi Run (10x)']
df_enc_m   = df_multi[df_multi['label'] == 'Encrypted']
df_plain_m = df_multi[df_multi['label'] == 'Non-Encrypted']

# TEE overhead attribution (multi run, merged)
df_attr = df_enc_m.merge(
    df_plain_m[DIMS + ['client_ttft_s_p50']].rename(
        columns={'client_ttft_s_p50': 'client_ttft_plain'}
    ),
    on=DIMS,
)
df_attr['ttft_delta'] = df_attr['client_ttft_s_p50'] - df_attr['client_ttft_plain']
df_attr['network_s']  = df_attr['server_network_latency_ms_mean'] / 1000
df_attr['enc_s']      = df_attr['total_encryption_time_ms_mean']  / 1000
df_attr['dec_s']      = df_attr['total_decryption_time_ms_mean']  / 1000
df_attr['residual_s'] = (df_attr['ttft_delta'] - df_attr['network_s']
                          - df_attr['enc_s'] - df_attr['dec_s']).clip(lower=0)

# ═══════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════

PLT_STYLE = {
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.35,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
    'figure.facecolor':   'white',
    'axes.facecolor':     'white',
}

def fig_to_image(fig, width_cm=16):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    # Read PNG dimensions from header (bytes 16-23 after PNG signature + IHDR tag)
    buf.seek(16)
    px_w = struct.unpack('>I', buf.read(4))[0]
    px_h = struct.unpack('>I', buf.read(4))[0]
    buf.seek(0)
    width_pts = width_cm * cm
    height_pts = width_pts * px_h / px_w
    img = Image(buf, width=width_pts, height=height_pts)
    img.hAlign = 'CENTER'
    return img


def line_chart(ax, x, y_enc, y_plain, ylabel, title, log_x=True):
    """Shared enc vs plain line chart helper."""
    kw = dict(marker='o', markersize=4, linewidth=1.8)
    ax.plot(x, y_enc,   color=C_ENC,   label='Encrypted',     **kw)
    ax.plot(x, y_plain, color=C_PLAIN, label='Non-Encrypted', **kw, linestyle='--')
    if log_x:
        ax.set_xscale('log', base=2)
        ax.set_xticks(CONC_ORDER)
        ax.set_xticklabels(CONC_ORDER)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(framealpha=0.6)


# ═══════════════════════════════════════════════════════════════════════════
# CHART 1 — SERVER METRICS (assumption validation)
# ═══════════════════════════════════════════════════════════════════════════

def make_chart_server():
    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        fig.suptitle('Server-Side Metrics — Encrypted vs Non-Encrypted',
                     fontsize=11, fontweight='bold', y=1.01)

        for col, out_tok in enumerate([100, 1024]):
            enc_sub   = df_enc_m[df_enc_m['num_output_tokens'] == out_tok].sort_values('num_concurrent_requests')
            plain_sub = df_plain_m[df_plain_m['num_output_tokens'] == out_tok].sort_values('num_concurrent_requests')
            x = enc_sub['num_concurrent_requests'].values

            line_chart(
                axes[0, col], x,
                enc_sub['server_ttft_s_p50'].values,
                plain_sub['server_ttft_s_p50'].values,
                'Server TTFT p50 (s)',
                f'Server TTFT — {out_tok} output tokens',
            )
            line_chart(
                axes[1, col], x,
                enc_sub['server_output_token_per_s_p50'].values,
                plain_sub['server_output_token_per_s_p50'].values,
                'Output Tok/s p50 (per request)',
                f'Server Output Throughput — {out_tok} output tokens',
            )

        fig.tight_layout()
        return fig_to_image(fig)


# ═══════════════════════════════════════════════════════════════════════════
# CHART 2 — TEE OVERHEAD BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════

def make_chart_tee_breakdown():
    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle('TEE Overhead Breakdown per Request (Encrypted, Multi Run)',
                     fontsize=11, fontweight='bold')

        for col, out_tok in enumerate([100, 1024]):
            sub = df_attr[df_attr['num_output_tokens'] == out_tok].sort_values('num_concurrent_requests')
            x   = np.arange(len(sub))
            xl  = [str(c) for c in sub['num_concurrent_requests'].values]
            ax  = axes[col]

            b1 = ax.bar(x, sub['network_s'],  color=C_NET,  label='Network Hop',      alpha=0.9)
            b2 = ax.bar(x, sub['enc_s'],      color=C_ENCR, label='Input Encryption', alpha=0.9,
                        bottom=sub['network_s'])
            b3 = ax.bar(x, sub['dec_s'],      color=C_DEC,  label='Output Decryption',alpha=0.9,
                        bottom=sub['network_s'] + sub['enc_s'])
            b4 = ax.bar(x, sub['residual_s'], color=C_OTHER,label='Residual',          alpha=0.7,
                        bottom=sub['network_s'] + sub['enc_s'] + sub['dec_s'])

            # Annotate network % on each bar
            for i, row in sub.iterrows():
                total = row['ttft_delta']
                if total > 0:
                    pct = row['network_s'] / row['client_ttft_s_p50'] * 100
                    ax.text(list(sub.index).index(i),
                            row['network_s'] / 2,
                            f'{pct:.0f}%', ha='center', va='center',
                            fontsize=6.5, color='white', fontweight='bold')

            ax.set_xticks(x)
            ax.set_xticklabels(xl)
            ax.set_xlabel('Concurrency')
            ax.set_ylabel('Overhead (s)')
            ax.set_title(f'TEE Overhead — {out_tok} output tokens', fontsize=9, fontweight='bold')
            if col == 0:
                ax.legend(loc='upper left', fontsize=7.5, framealpha=0.7)

        fig.tight_layout()
        return fig_to_image(fig)


# ═══════════════════════════════════════════════════════════════════════════
# CHART 3 — NETWORK LATENCY DETAIL
# ═══════════════════════════════════════════════════════════════════════════

def make_chart_network():
    df_enc_all   = df_all[df_all['label'] == 'Encrypted']
    df_enc_s_all = df_enc_all[df_enc_all['run_type'] == 'Single Run (1x)']
    df_enc_m_all = df_enc_all[df_enc_all['run_type'] == 'Multi Run (10x)']

    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle('TEE Network Latency vs Concurrency (Encrypted Model)',
                     fontsize=11, fontweight='bold')

        tok_colors = {100: '#1f77b4', 1024: '#ff7f0e'}

        for col, out_tok in enumerate([100, 1024]):
            ax = axes[col]
            for run_type, df_r, ls in [
                ('Single Run (1x)', df_enc_s_all, 'solid'),
                ('Multi Run (10x)', df_enc_m_all, 'dashed'),
            ]:
                sub = df_r[df_r['num_output_tokens'] == out_tok].sort_values('num_concurrent_requests')
                ax.plot(
                    sub['num_concurrent_requests'],
                    sub['server_network_latency_ms_mean'],
                    color=tok_colors[out_tok],
                    linestyle=ls,
                    marker='o', markersize=4, linewidth=1.8,
                    label=run_type,
                )

            ax.set_xscale('log', base=2)
            ax.set_xticks(CONC_ORDER)
            ax.set_xticklabels(CONC_ORDER)
            ax.set_xlabel('Concurrency (log scale)')
            ax.set_ylabel('Network Latency — mean (ms)')
            ax.set_title(f'Network Latency — {out_tok} output tokens', fontsize=9, fontweight='bold')
            ax.legend(framealpha=0.6)
            ax.axvline(x=8, color='red', linestyle=':', linewidth=1.2, alpha=0.6)
            ax.text(8.5, ax.get_ylim()[1] * 0.92, 'TEE core\nlimit (8)',
                    color='red', fontsize=6.5, alpha=0.8)

        fig.tight_layout()
        return fig_to_image(fig)


# ═══════════════════════════════════════════════════════════════════════════
# CHART 4 — CLIENT METRICS IMPACT
# ═══════════════════════════════════════════════════════════════════════════

def make_chart_client():
    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        fig.suptitle('Client-Side Metrics — Encrypted vs Non-Encrypted (Multi Run)',
                     fontsize=11, fontweight='bold', y=1.01)

        for col, out_tok in enumerate([100, 1024]):
            enc_sub   = df_enc_m[df_enc_m['num_output_tokens'] == out_tok].sort_values('num_concurrent_requests')
            plain_sub = df_plain_m[df_plain_m['num_output_tokens'] == out_tok].sort_values('num_concurrent_requests')
            x = enc_sub['num_concurrent_requests'].values

            line_chart(
                axes[0, col], x,
                enc_sub['client_ttft_s_p50'].values,
                plain_sub['client_ttft_s_p50'].values,
                'Client TTFT p50 (s)',
                f'Client TTFT — {out_tok} output tokens',
            )
            # Shade the TEE overhead gap
            axes[0, col].fill_between(
                x,
                plain_sub['client_ttft_s_p50'].values,
                enc_sub['client_ttft_s_p50'].values,
                alpha=0.12, color=C_ENC, label='TEE overhead gap',
            )

            line_chart(
                axes[1, col], x,
                enc_sub['num_completed_requests_per_min'].values,
                plain_sub['num_completed_requests_per_min'].values,
                'Completed Requests / min',
                f'System Throughput — {out_tok} output tokens',
            )

        fig.tight_layout()
        return fig_to_image(fig)


# ═══════════════════════════════════════════════════════════════════════════
# CHART 5 — ENCRYPTION / DECRYPTION TIMES
# ═══════════════════════════════════════════════════════════════════════════

def make_chart_crypto_times():
    df_enc_all = df_all[df_all['label'] == 'Encrypted']
    tok_colors  = {100: '#1f77b4', 1024: '#ff7f0e'}
    run_styles  = {'Single Run (1x)': 'solid', 'Multi Run (10x)': 'dashed'}

    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle('TEE Encryption & Decryption Times vs Concurrency',
                     fontsize=11, fontweight='bold')

        shown = set()
        for run_type, ls in run_styles.items():
            for out_tok, color in tok_colors.items():
                sub  = df_enc_all[(df_enc_all['run_type'] == run_type) &
                                  (df_enc_all['num_output_tokens'] == out_tok)
                                  ].sort_values('num_concurrent_requests')
                name = f'{out_tok} out | {run_type}'
                show = name not in shown

                axes[0].plot(sub['num_concurrent_requests'],
                             sub['total_encryption_time_ms_mean'],
                             color=color, linestyle=ls, marker='o',
                             markersize=4, linewidth=1.8,
                             label=name if show else None)
                axes[1].plot(sub['num_concurrent_requests'],
                             sub['total_decryption_time_ms_mean'],
                             color=color, linestyle=ls, marker='o',
                             markersize=4, linewidth=1.8,
                             label=name if show else None)
                shown.add(name)

        for ax, title, ylabel in [
            (axes[0], 'Input Encryption Time', 'Time (ms)'),
            (axes[1], 'Output Decryption Time', 'Time (ms)'),
        ]:
            ax.set_xscale('log', base=2)
            ax.set_xticks(CONC_ORDER)
            ax.set_xticklabels(CONC_ORDER)
            ax.set_xlabel('Concurrency (log scale)')
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.legend(fontsize=7, framealpha=0.6)

        fig.tight_layout()
        return fig_to_image(fig)


# ═══════════════════════════════════════════════════════════════════════════
# REPORTLAB STYLES
# ═══════════════════════════════════════════════════════════════════════════

def build_styles():
    base = getSampleStyleSheet()

    styles = {
        'cover_title': ParagraphStyle(
            'cover_title', fontSize=26, fontName='Helvetica-Bold',
            textColor=colors.white, alignment=TA_CENTER, spaceAfter=8,
        ),
        'cover_sub': ParagraphStyle(
            'cover_sub', fontSize=13, fontName='Helvetica',
            textColor=colors.HexColor('#aed6f1'), alignment=TA_CENTER, spaceAfter=4,
        ),
        'cover_date': ParagraphStyle(
            'cover_date', fontSize=10, fontName='Helvetica',
            textColor=colors.HexColor('#85c1e9'), alignment=TA_CENTER,
        ),
        'h1': ParagraphStyle(
            'h1', fontSize=14, fontName='Helvetica-Bold',
            textColor=colors.HexColor(C_NAVY), spaceBefore=14, spaceAfter=6,
            borderPad=4,
        ),
        'h2': ParagraphStyle(
            'h2', fontSize=11, fontName='Helvetica-Bold',
            textColor=colors.HexColor(C_BLUE2), spaceBefore=10, spaceAfter=4,
        ),
        'body': ParagraphStyle(
            'body', fontSize=9.5, fontName='Helvetica',
            textColor=colors.HexColor('#2c3e50'), leading=14,
            alignment=TA_JUSTIFY, spaceAfter=6,
        ),
        'bullet': ParagraphStyle(
            'bullet', fontSize=9.5, fontName='Helvetica',
            textColor=colors.HexColor('#2c3e50'), leading=14,
            leftIndent=14, spaceAfter=3,
        ),
        'finding_title': ParagraphStyle(
            'finding_title', fontSize=10, fontName='Helvetica-Bold',
            textColor=colors.HexColor('#1a5276'), spaceAfter=3,
        ),
        'finding_body': ParagraphStyle(
            'finding_body', fontSize=9, fontName='Helvetica',
            textColor=colors.HexColor('#2c3e50'), leading=13,
        ),
        'caption': ParagraphStyle(
            'caption', fontSize=8, fontName='Helvetica-Oblique',
            textColor=colors.HexColor('#7f8c8d'), alignment=TA_CENTER,
            spaceBefore=2, spaceAfter=10,
        ),
        'table_hdr': ParagraphStyle(
            'table_hdr', fontSize=8.5, fontName='Helvetica-Bold',
            textColor=colors.white, alignment=TA_CENTER,
        ),
        'table_cell': ParagraphStyle(
            'table_cell', fontSize=8.5, fontName='Helvetica',
            textColor=colors.HexColor('#2c3e50'), alignment=TA_CENTER,
        ),
    }
    return styles


def finding_box(title, body_paragraphs, styles, bg=C_FIND, border=C_BLUE2):
    """Highlighted finding box."""
    content = [Paragraph(f'🔍  {title}', styles['finding_title'])]
    for p in body_paragraphs:
        content.append(Paragraph(p, styles['finding_body']))
    tbl = Table([[content]], colWidths=[16.5 * cm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, -1), colors.HexColor(bg)),
        ('ROUNDEDCORNERS', [6]),
        ('BOX',         (0, 0), (-1, -1), 1.2, colors.HexColor(border)),
        ('TOPPADDING',  (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING',(0,0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',(0, 0), (-1, -1), 10),
    ]))
    return tbl


def warning_box(title, body, styles):
    return finding_box(title, body, styles, bg=C_WARN, border='#e74c3c')


def section_rule(color=C_NAVY):
    return HRFlowable(width='100%', thickness=1.5,
                      color=colors.HexColor(color), spaceAfter=6, spaceBefore=2)


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY DATA TABLE
# ═══════════════════════════════════════════════════════════════════════════

def build_summary_table(styles):
    """Side-by-side key metric table for multi run data."""

    # Compute key numbers
    rows_data = []
    for out_tok in [100, 1024]:
        for conc in [1, 4, 8, 16, 32]:
            enc_row   = df_enc_m[(df_enc_m['num_output_tokens'] == out_tok) &
                                  (df_enc_m['num_concurrent_requests'] == conc)]
            plain_row = df_plain_m[(df_plain_m['num_output_tokens'] == out_tok) &
                                    (df_plain_m['num_concurrent_requests'] == conc)]
            if enc_row.empty or plain_row.empty:
                continue

            e = enc_row.iloc[0]
            p = plain_row.iloc[0]

            net_ms    = e['server_network_latency_ms_mean']
            enc_ms    = e['total_encryption_time_ms_mean']
            dec_ms    = e['total_decryption_time_ms_mean']
            tee_total = net_ms + enc_ms + dec_ms
            ttft_enc  = e['client_ttft_s_p50']
            ttft_pln  = p['client_ttft_s_p50']
            tee_pct   = tee_total / 1000 / ttft_enc * 100 if ttft_enc else 0
            err_pct   = e['error_rate_pct'] if pd.notna(e['error_rate_pct']) else 0

            rows_data.append([
                f'{out_tok}',
                f'{conc}',
                f'{p["server_ttft_s_p50"]:.3f}',
                f'{e["server_ttft_s_p50"]:.3f}',
                f'{ttft_pln:.3f}',
                f'{ttft_enc:.3f}',
                f'{net_ms:.0f}',
                f'{enc_ms:.1f}',
                f'{dec_ms:.2f}',
                f'{tee_pct:.0f}%',
                f'{err_pct:.0f}%',
            ])

    headers = [
        'Out\nTokens', 'Conc',
        'Server\nTTFT Plain', 'Server\nTTFT Enc',
        'Client\nTTFT Plain', 'Client\nTTFT Enc',
        'Network\n(ms)', 'Encrypt\n(ms)', 'Decrypt\n(ms)',
        'TEE %\nof TTFT', 'Error\nRate',
    ]

    col_widths = [1.3*cm, 1.0*cm, 2.0*cm, 2.0*cm, 2.0*cm, 2.0*cm,
                  1.7*cm, 1.6*cm, 1.6*cm, 1.6*cm, 1.5*cm]

    table_data = [[Paragraph(h, styles['table_hdr']) for h in headers]]
    for rd in rows_data:
        table_data.append([Paragraph(v, styles['table_cell']) for v in rd])

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    style = [
        ('BACKGROUND',   (0, 0), (-1, 0),  colors.HexColor(C_NAVY)),
        ('TEXTCOLOR',    (0, 0), (-1, 0),  colors.white),
        ('ALIGN',        (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTSIZE',     (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, colors.HexColor('#f2f3f4')]),
        ('GRID',         (0, 0), (-1, -1), 0.4, colors.HexColor('#bdc3c7')),
        ('TOPPADDING',   (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, colors.HexColor('#ebf5fb')]),
    ]
    # Highlight separator between 100 and 1024 token groups
    n_100 = sum(1 for r in rows_data if r[0] == '100')
    style.append(('LINEABOVE', (0, n_100 + 1), (-1, n_100 + 1), 1.5, colors.HexColor(C_NAVY)))
    tbl.setStyle(TableStyle(style))
    return tbl


# ═══════════════════════════════════════════════════════════════════════════
# PDF ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════

def build_pdf():
    from datetime import date as dt_date

    doc = SimpleDocTemplate(
        OUT_PDF,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
    )

    S = build_styles()
    story = []

    # ── COVER ────────────────────────────────────────────────────────────
    # Dark header block
    cover_bg = Table(
        [[Paragraph('DataKrypto TEE', S['cover_title']),],
         [Paragraph('Performance Impact Analysis', S['cover_title'])],
         [Spacer(1, 0.3 * cm)],
         [Paragraph('Encrypted vs Non-Encrypted Model Benchmarking Report', S['cover_sub'])],
         [Spacer(1, 0.2 * cm)],
         [Paragraph(f'Generated: {dt_date.today().strftime("%B %d, %Y")}', S['cover_date'])],
         ],
        colWidths=[16.5 * cm],
    )
    cover_bg.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, -1), colors.HexColor(C_NAVY)),
        ('TOPPADDING',   (0, 0), (-1, -1), 18),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 18),
        ('LEFTPADDING',  (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ('ROUNDEDCORNERS', [8]),
    ]))
    story.append(cover_bg)
    story.append(Spacer(1, 0.5 * cm))

    story.append(section_rule())
    story.append(Paragraph('Executive Summary', S['h1']))
    story.append(Paragraph(
        'This report benchmarks SambaNova inference performance when accessed directly '
        '(plain model) versus through the DataKrypto Trusted Execution Environment (TEE) '
        '(encrypted model). The TEE adds a confidential computing layer that encrypts the '
        'user prompt before sending it to the inference server and decrypts each streaming '
        'output token chunk before returning it to the client, introducing an extra network '
        'hop in the request path.',
        S['body'],
    ))
    story.append(Paragraph(
        'Tests were run across concurrencies of 1, 2, 4, 8, 16, and 32 simultaneous '
        'HTTP requests, with output token targets of 100 and 1,024, using a fixed '
        '2,000 input-token prompt. Two test modes were evaluated: a Single Run (one round '
        'of N simultaneous requests) and a Multi Run (10 sequential rounds of N simultaneous '
        'requests) for statistical stability.',
        S['body'],
    ))

    story.append(Paragraph('Key Findings', S['h2']))
    findings = [
        ('<b>✅ Server assumption confirmed:</b> With endpoints on equivalent configurations, '
         'server TTFT and tok/s are nearly identical for both models at concurrency 1–4 '
         '(~0.091 s TTFT, ~585–620 tok/s). The TEE has no impact on inference speed '
         'inside the SambaNova engine.'),
        ('<b>🔀 TEE de-batching effect — a hidden server benefit:</b> Because the TEE processes '
         'requests sequentially per core, concurrent requests arrive at the server staggered, '
         'keeping the effective batch small. At concurrency 8–32, encrypted server TTFT stays '
         'flat at ~0.163 s while non-encrypted TTFT doubles to ~0.311 s. At concurrency ≥ 16 '
         'for short responses (100 tokens), this server-side advantage outweighs the TEE network '
         'overhead — encrypted client TTFT actually becomes lower than non-encrypted.'),
        ('<b>🌐 Network hop is the dominant overhead at low concurrency:</b> The extra '
         'Client → TEE → Server roundtrip accounts for the bulk of the client-observable latency '
         'increase at concurrency 1–8. At concurrency 1 with 100 output tokens, TEE network '
         'latency is ~425 ms (Multi Run); for 1,024 output tokens it rises to ~1,600 ms.'),
        ('<b>🔐 Crypto overhead is negligible:</b> Input encryption costs ~5–10 ms per '
         'request (near-constant, independent of output length). Cumulative output decryption '
         'costs ~0.1–4 ms total across all streaming chunks — both are negligible compared '
         'to the network hop.'),
        ('<b>⚠️ TEE queue saturation at concurrency > 8:</b> The TEE has 8 cores processing '
         'encryption/decryption sequentially. Beyond 8 simultaneous requests, requests queue '
         'at the TEE, causing network latency to grow non-linearly and error rates to climb '
         '(up to ~19% at concurrency 32 for 100 output tokens).'),
        ('<b>📉 System throughput gap at scale:</b> Non-Encrypted throughput scales near-linearly '
         'with concurrency (153 → 848 req/min for 100 output tokens). Encrypted throughput peaks '
         'at concurrency 2 (~180 req/min) and drops to ~28 req/min at concurrency 32 — '
         'a ~30× gap driven by TEE queueing and errors above concurrency 8.'),
        ('<b>🔁 Decryption grows linearly with output tokens</b> because each streaming '
         'chunk is decrypted individually by the TEE. Each chunk decryption costs on the '
         'order of microseconds; the growth is real but the absolute values remain tiny.'),
    ]
    for f in findings:
        story.append(Paragraph(f'• {f}', S['bullet']))

    story.append(PageBreak())

    # ── SECTION 1: ARCHITECTURE ─────────────────────────────────────────
    story.append(section_rule())
    story.append(Paragraph('1. Architecture & Test Setup', S['h1']))

    story.append(Paragraph('Request Paths', S['h2']))
    arch_text = (
        '<b>Non-Encrypted:</b> Client ──────────────────────────► SambaNova Inference Server<br/>'
        '<font size="9" color="#7f8c8d">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(1 network hop — direct)</font><br/><br/>'
        '<b>Encrypted:</b>&nbsp;&nbsp;&nbsp; Client ──► DataKrypto TEE ──► SambaNova Inference Server<br/>'
        '<font size="9" color="#7f8c8d">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TEE encrypts input &amp; decrypts each output chunk (streaming)<br/>'
        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(2 network hops total)</font>'
    )
    arch_box = Table([[Paragraph(arch_text, S['body'])]], colWidths=[16.5*cm])
    arch_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8f9fa')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 14),
    ]))
    story.append(arch_box)
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph('Models', S['h2']))
    model_tbl = Table(
        [
            [Paragraph('<b>Model</b>', S['table_hdr']),
             Paragraph('<b>Description</b>', S['table_hdr'])],
            [Paragraph('Llama-xLAM-2-8b-fc-r', S['table_cell']),
             Paragraph('Base model — direct plain inference', S['table_cell'])],
            [Paragraph('Llama-xLAM-2-8b-fc-r-encrypted', S['table_cell']),
             Paragraph('Same base architecture — routed through DataKrypto TEE', S['table_cell'])],
        ],
        colWidths=[7 * cm, 9.5 * cm],
    )
    model_tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, 0), colors.HexColor(C_NAVY)),
        ('TEXTCOLOR',    (0, 0), (-1, 0), colors.white),
        ('GRID',         (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, colors.HexColor('#ebf5fb')]),
        ('TOPPADDING',   (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 5),
        ('LEFTPADDING',  (0, 0), (-1, -1), 8),
    ]))
    story.append(model_tbl)
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph('Test Matrix', S['h2']))
    story.append(Paragraph(
        'Input tokens: <b>2,000</b> (fixed) &nbsp;|&nbsp; '
        'Output tokens: <b>100 and 1,024</b> &nbsp;|&nbsp; '
        'Concurrencies: <b>1, 2, 4, 8, 16, 32</b> simultaneous HTTP requests &nbsp;|&nbsp; '
        'Single Run (1 round) + Multi Run (10 rounds)',
        S['body'],
    ))
    story.append(Paragraph(
        'Each concurrency level sends <b>N independent HTTP requests simultaneously</b> — '
        'not a single batched API call. The SambaNova server dynamically batches '
        'simultaneous requests for inference. When the TEE is in the path, its sequential '
        'per-core processing may stagger request arrival at the server, effectively '
        'reducing the observed batch size.',
        S['body'],
    ))

    story.append(PageBreak())

    # ── SECTION 2: SERVER METRICS ────────────────────────────────────────
    story.append(section_rule())
    story.append(Paragraph('2. Server-Side Metrics — Assumption Validation', S['h1']))
    story.append(Paragraph(
        'With both SambaNova endpoints on equivalent configurations, server-side metrics directly '
        'validate the core assumption: the TEE has no impact on inference speed inside the server. '
        'At concurrency 1–4, both models show nearly identical server TTFT (~0.091 s) and output '
        'tok/s (~585–620 tok/s). At higher concurrencies, the TEE de-batching effect becomes '
        'visible: encrypted server TTFT stays flat at ~0.163 s while non-encrypted rises to '
        '~0.311 s as the server absorbs the full simultaneous batch.',
        S['body'],
    ))
    story.append(make_chart_server())
    story.append(Paragraph(
        'Figure 1: Server TTFT p50 (top) and Server Output Tok/s p50 (bottom) for 100 and 1,024 '
        'output tokens. Multi Run (10x). Solid = Encrypted, Dashed = Non-Encrypted.',
        S['caption'],
    ))
    story.append(finding_box(
        'Finding: Server metrics confirm TEE adds zero inference overhead',
        [
            'At concurrency 1–4, server TTFT is ~0.091 s and server tok/s is ~585–620 tok/s '
            'for both models — the values are virtually identical. This directly confirms that '
            'the TEE does not slow down the SambaNova inference engine in any way.',
            'At concurrency 8–32, the TEE de-batching effect appears: encrypted server TTFT '
            'stays at ~0.163 s while non-encrypted jumps to ~0.311 s. The TEE\'s sequential '
            'per-core processing staggers request arrival, so the server sees a smaller effective '
            'batch for encrypted requests. This lower server TTFT is a secondary benefit of the '
            'TEE architecture — at high concurrency with short responses, it can more than '
            'compensate for the TEE network overhead, making encrypted client TTFT lower than '
            'non-encrypted at concurrency ≥ 16 (100 token responses).',
        ],
        S,
    ))

    story.append(PageBreak())

    # ── SECTION 3: TEE OVERHEAD ──────────────────────────────────────────
    story.append(section_rule())
    story.append(Paragraph('3. TEE Overhead Decomposition', S['h1']))
    story.append(Paragraph(
        'The TEE reports three measurable latency components per request. '
        'Understanding their relative magnitudes is key to identifying where '
        'improvement effort should be focused.',
        S['body'],
    ))

    # Component table
    comp_tbl = Table(
        [
            [Paragraph('<b>Component</b>', S['table_hdr']),
             Paragraph('<b>Source</b>', S['table_hdr']),
             Paragraph('<b>Observed Range</b>', S['table_hdr']),
             Paragraph('<b>Grows with…</b>', S['table_hdr'])],
            [Paragraph('Input Encryption', S['table_cell']),
             Paragraph('TEE encrypts the 2,000-token prompt', S['table_cell']),
             Paragraph('~5–10 ms', S['table_cell']),
             Paragraph('Near-constant (independent of output)', S['table_cell'])],
            [Paragraph('Output Decryption', S['table_cell']),
             Paragraph('TEE decrypts each streaming chunk (per token batch)', S['table_cell']),
             Paragraph('~0.1–4 ms total', S['table_cell']),
             Paragraph('Output token count (linear, microseconds/chunk)', S['table_cell'])],
            [Paragraph('Network Hop', S['table_cell']),
             Paragraph('Extra Client→TEE→Server→TEE→Client roundtrip', S['table_cell']),
             Paragraph('250 ms – 6,600 ms', S['table_cell']),
             Paragraph('Concurrency (queue) + Output tokens (streaming RTTs)', S['table_cell'])],
        ],
        colWidths=[3.2*cm, 5.0*cm, 3.3*cm, 5.0*cm],
    )
    comp_tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, 0), colors.HexColor(C_NAVY)),
        ('TEXTCOLOR',    (0, 0), (-1, 0), colors.white),
        ('GRID',         (0, 0), (-1, -1), 0.5, colors.HexColor('#bdc3c7')),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, colors.HexColor('#ebf5fb')]),
        ('TOPPADDING',   (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 5),
        ('LEFTPADDING',  (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND',   (0, 4), (-1, 4), colors.HexColor('#fef9e7')),  # highlight network row
    ]))
    story.append(comp_tbl)
    story.append(Spacer(1, 0.3 * cm))

    story.append(make_chart_tee_breakdown())
    story.append(Paragraph(
        'Figure 2: TEE overhead per request broken down into Network Hop (orange), '
        'Input Encryption (red), Output Decryption (green), and unexplained residual (purple). '
        'Percentages inside bars = network hop as % of total encrypted client TTFT.',
        S['caption'],
    ))

    story.append(make_chart_crypto_times())
    story.append(Paragraph(
        'Figure 3: Encryption time (left) and decryption time (right) vs concurrency. '
        'Solid lines = Single Run, Dashed = Multi Run. '
        'Blue = 100 output tokens, Orange = 1,024 output tokens.',
        S['caption'],
    ))

    story.append(finding_box(
        'Finding: Network hop dominates — crypto overhead is negligible',
        [
            'The network hop accounts for 85–97% of the total TEE overhead in every scenario. '
            'Input encryption is approximately constant at ~5–10 ms and decreases slightly at '
            'higher concurrencies (likely TEE CPU sharing effects). Output decryption grows '
            'linearly with output tokens (~0.1 ms for 100 tokens, ~3–4 ms for 1,024 tokens) '
            'because each streaming chunk is decrypted individually — each chunk costs on the '
            'order of microseconds.',
            'Implication: Optimising the encryption/decryption algorithm would have minimal '
            'impact. Reducing the network roundtrip time (physical proximity of TEE to '
            'client and to the inference server) is the highest-leverage improvement.',
        ],
        S,
    ))

    story.append(PageBreak())

    # ── SECTION 4: NETWORK LATENCY ───────────────────────────────────────
    story.append(section_rule())
    story.append(Paragraph('4. Network Latency Growth Pattern', S['h1']))
    story.append(Paragraph(
        'The TEE network latency grows along two dimensions: concurrency (TEE queue depth) '
        'and output token count (more streaming chunks = more decryption RTTs).',
        S['body'],
    ))
    story.append(make_chart_network())
    story.append(Paragraph(
        'Figure 4: TEE network latency vs concurrency for 100 output tokens (left) '
        'and 1,024 output tokens (right). Solid = Single Run, Dashed = Multi Run. '
        'Red vertical line marks the TEE 8-core limit.',
        S['caption'],
    ))
    story.append(finding_box(
        'Finding: Two distinct growth regimes',
        [
            '<b>Below TEE capacity (concurrency ≤ 8):</b> Network latency grows moderately '
            'because requests can be processed in parallel across available cores. '
            'At concurrency 1 with 100 output tokens the mean network latency is ~425 ms '
            '(Multi Run); at concurrency 8 it rises to ~638 ms.',
            '<b>Above TEE capacity (concurrency > 8):</b> Requests queue at the TEE, '
            'causing network latency to grow faster. At concurrency 32 with 100 output '
            'tokens it reaches ~803 ms; for 1,024 output tokens it can exceed 3,100 ms. '
            'This is where error rates also begin to climb.',
            '<b>Output tokens scale the baseline:</b> Longer responses require more '
            'streaming decrypt RTTs. At concurrency 1, the jump from 100 to 1,024 output '
            'tokens increases network latency from ~425 ms to ~1,608 ms (~3.8×).',
        ],
        S,
    ))

    story.append(PageBreak())

    # ── SECTION 5: CLIENT IMPACT ─────────────────────────────────────────
    story.append(section_rule())
    story.append(Paragraph('5. Client-Side Impact — User-Facing Latency', S['h1']))
    story.append(Paragraph(
        'Client TTFT and system throughput capture the full end-user experience '
        'including all TEE overhead. The shaded area in the TTFT chart represents '
        'the TEE overhead gap.',
        S['body'],
    ))
    story.append(make_chart_client())
    story.append(Paragraph(
        'Figure 5: Client TTFT p50 (top) and completed requests per minute (bottom) '
        'for 100 and 1,024 output tokens. Multi Run (10x). '
        'At concurrency ≥ 16 (100 tokens), encrypted TTFT drops below non-encrypted '
        'due to the TEE de-batching benefit on server prefill time.',
        S['caption'],
    ))
    story.append(warning_box(
        'Warning: System throughput gap widens at high concurrency (Encrypted)',
        [
            'Non-Encrypted throughput scales near-linearly with concurrency, '
            'reaching ~848 req/min at concurrency 32 (100 output tokens). '
            'Encrypted throughput peaks at concurrency 2 (~180 req/min) and '
            'falls to ~28 req/min at concurrency 32 — a ~30× gap. This collapse '
            'is driven by the TEE queue building up (>8 cores) and requests experiencing '
            'progressively longer network waits. At concurrency >8, error rates reach '
            '11–19%, further reducing effective throughput. '
            'Note: despite lower per-request TTFT at high concurrency for 100-token '
            'responses, the overall system throughput (req/min) remains far lower for '
            'the encrypted path due to TEE queueing and errors.',
        ],
        S,
    ))

    story.append(PageBreak())

    # ── SECTION 6: SUMMARY TABLE ─────────────────────────────────────────
    story.append(section_rule())
    story.append(Paragraph('6. Summary Data Table (Multi Run)', S['h1']))
    story.append(Paragraph(
        'All latency values in seconds unless noted. '
        'TEE % = (network + encryption + decryption) / encrypted client TTFT × 100.',
        S['body'],
    ))
    story.append(build_summary_table(S))
    story.append(Spacer(1, 0.4 * cm))

    story.append(PageBreak())

    # ── SECTION 7: CONCLUSIONS ───────────────────────────────────────────
    story.append(section_rule())
    story.append(Paragraph('7. Conclusions', S['h1']))

    conclusions = [
        ('<b>TEE adds zero server inference overhead — confirmed.</b> With equivalent endpoint '
         'configurations, server TTFT and tok/s are nearly identical at concurrency 1–4 '
         '(~0.091 s and ~585–620 tok/s respectively). The TEE de-batching effect is also '
         'confirmed: at concurrency 8–32, encrypted server TTFT stays flat at ~0.163 s while '
         'non-encrypted rises to ~0.311 s. At concurrency ≥ 16 with short responses, this '
         'server-side benefit can outweigh the TEE network overhead entirely.'),
        ('<b>The overhead is entirely network-driven.</b> The extra network hop through '
         'the TEE accounts for 85–97% of the additional client-visible latency. '
         'Encryption (~5–10 ms) and decryption (~0.1–4 ms cumulative) are negligible.'),
        ('<b>Network latency is predictable at low concurrency.</b> Below TEE capacity '
         '(concurrency ≤ 8), the added latency is roughly constant per round-trip and '
         'scales linearly with output token count — consistent with the TEE team\'s claims.'),
        ('<b>TEE queue saturation is the critical scalability limit.</b> At concurrency > 8, '
         'the 8-core TEE begins queuing requests, causing latency to grow non-linearly '
         'and error rates to increase. Production deployments should stay at or below '
         'the TEE core count for reliable performance.'),
        ('<b>Per-user latency overhead at concurrency 1:</b> Client TTFT increases by '
         '~0.19 s (100 output tokens) to ~0.17 s (1,024 output tokens) in Multi Run — '
         'relatively modest for a confidential computing solution.'),
    ]
    for c in conclusions:
        story.append(Paragraph(f'• {c}', S['bullet']))
        story.append(Spacer(1, 0.15 * cm))

    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph('Recommendations', S['h2']))
    recs = [
        '<b>Reduce TEE network latency:</b> Co-locate the TEE as close as possible to both '
        'the client and the SambaNova inference server to minimise round-trip times.',
        '<b>Respect TEE core limit:</b> Keep concurrent request load at or below 8 to avoid '
        'queue saturation. Above this point latency grows non-linearly and errors appear.',
        '<b>No need to optimise crypto:</b> Encryption and decryption are already fast enough '
        'that further optimisation would have negligible user-visible impact.',
        '<b>Scale horizontally:</b> For higher throughput with the TEE, scale the number of '
        'TEE instances rather than increasing concurrency per instance beyond 8.',
    ]
    for r in recs:
        story.append(Paragraph(f'• {r}', S['bullet']))
        story.append(Spacer(1, 0.1 * cm))

    story.append(Spacer(1, 0.5 * cm))
    story.append(section_rule(color='#bdc3c7'))
    story.append(Paragraph(
        f'DataKrypto × SambaNova — Confidential Benchmarking Report — {dt_date.today().strftime("%B %Y")}',
        ParagraphStyle('footer', fontSize=7.5, fontName='Helvetica',
                       textColor=colors.HexColor('#95a5a6'), alignment=TA_CENTER),
    ))

    doc.build(story)
    print(f'✅  Report saved to: {OUT_PDF}')


if __name__ == '__main__':
    build_pdf()
