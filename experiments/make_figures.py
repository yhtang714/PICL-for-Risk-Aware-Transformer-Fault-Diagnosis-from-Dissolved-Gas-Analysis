"""Figures of the paper from the CSV files in tables/.

python experiments/make_figures.py --tables tables --out figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
TXT = '#0b0b0b'
TXT2 = '#52514e'
GRID = '#e6e5e1'
FAULTS = ['PD', 'D1', 'D2', 'T1', 'T2', 'T3']

plt.rcParams.update(
    {
        'font.family': 'serif',
        'font.size': 9,
        'axes.edgecolor': TXT2,
        'axes.labelcolor': TXT,
        'xtick.color': TXT2,
        'ytick.color': TXT2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': GRID,
        'grid.linewidth': 0.6,
        'axes.axisbelow': True,
        'legend.frameon': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    }
)


def _save(fig, out, name):
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(out / f'{name}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)


def fig_risk_coverage(tables, out):
    df = pd.read_csv(tables / 'risk_coverage_curves_all.csv')
    keys = [
        ('picl', 'PICL', PAL[0], '-'),
        ('picl_head', 'PICL without evidence fusion', PAL[0], '--'),
        ('random_forest', 'Random forest (gases + ratios)', PAL[1], '-'),
        ('xgboost', 'XGBoost (gases + ratios)', PAL[2], '-'),
        ('svm', 'SVM (gases + ratios)', PAL[3], '-'),
        ('random_forest_gases_only', 'Random forest (gases only)', PAL[1], ':'),
    ]
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    for key, label, col, ls in keys:
        d = df[df.key == key]
        g = d.groupby('coverage').risk
        m, lo, hi = g.mean(), g.quantile(0.1), g.quantile(0.9)
        ax.plot(m.index, m.values, color=col, ls=ls, lw=1.6, label=label)
        if key == 'picl':
            ax.fill_between(m.index, lo.values, hi.values, color=col, alpha=0.12, lw=0)
    ax.set_xlabel('Coverage (fraction of test records accepted)')
    ax.set_ylabel('Risk on accepted records (error rate)')
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7, loc='lower right', ncol=1)
    _save(fig, out, 'fig_risk_coverage')


def fig_matched_coverage(tables, out):
    df = pd.read_csv(tables / 'matched_coverage_summary.csv')
    keys = [
        ('picl', 'PICL', PAL[0], '-'),
        ('picl_head', 'PICL without evidence fusion', PAL[0], '--'),
        ('random_forest', 'Random forest (gases + ratios)', PAL[1], '-'),
        ('xgboost', 'XGBoost (gases + ratios)', PAL[2], '-'),
        ('random_forest_gases_only', 'Random forest (gases only)', PAL[1], ':'),
    ]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for key, label, col, ls in keys:
        d = df[df.key == key].sort_values('target_coverage')
        ax.plot(d.target_coverage, d.matched_acc, color=col, ls=ls, lw=1.6, marker='o', ms=3.5, label=label)
    ax.set_xlabel('Matched coverage')
    ax.set_ylabel('Accuracy on accepted records')
    ax.legend(fontsize=7.5)
    _save(fig, out, 'fig_matched_coverage')


def fig_per_class(tables, out):
    pc = pd.read_csv(tables / 'per_class_metrics.csv')
    methods = [
        ('PICL (all samples)', 'PICL', PAL[0]),
        ('Random forest', 'Random forest', PAL[1]),
        ('XGBoost', 'XGBoost', PAL[2]),
    ]
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    w = 0.26
    for i, (m, lab, col) in enumerate(methods):
        d = pc[pc.method == m].groupby('fault').recall.agg(['mean', 'std']).reindex(FAULTS)
        x = np.arange(6) + (i - 1) * w
        ax.bar(
            x,
            d['mean'],
            width=w - 0.03,
            color=col,
            label=lab,
            yerr=d['std'],
            error_kw=dict(elinewidth=0.8, ecolor=TXT2, capsize=0),
        )
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(FAULTS)
    ax.set_ylabel('Recall (mean over splits)')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15))
    _save(fig, out, 'fig_per_class_recall')


def fig_loso(tables, out):
    p = tables / 'v4_loso_per_fold.csv'
    if not p.exists():
        return
    df = pd.read_csv(p).groupby(['method', 'held_out'])[['macro_f1', 'fam_acc']].mean().reset_index()
    folds = ['IEC TC 10', 'IEEE DataPort', 'Published cases', 'NCEPR', 'NE Grid+Fujian']
    methods = [
        ('PICL, unseen-source read-out', 'PICL, unseen-source read-out', PAL[0]),
        ('PICL in-distribution read-out', 'PICL, in-distribution read-out', PAL[6]),
        ('Random forest', 'Random forest', PAL[1]),
        ('XGBoost', 'XGBoost', PAL[2]),
        ('SVM', 'SVM', PAL[4]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    short = {
        'IEC TC 10': 'TC 10',
        'IEEE DataPort': 'DataPort',
        'NCEPR': 'NCEPR',
        'NE Grid+Fujian': 'NE Grid\n+Fujian',
        'Published cases': 'Published',
    }
    for ax, col_, ylabel, ylim in (
        (axes[0], 'macro_f1', 'Sub-class macro-F1, held-out source', (0, 0.65)),
        (axes[1], 'fam_acc', 'Family accuracy, held-out source', (0.6, 1.0)),
    ):
        w = 0.16
        for i, (m, lab, col) in enumerate(methods):
            d = df[df.method == m].set_index('held_out').reindex(folds)
            ax.bar(np.arange(len(folds)) + (i - 2) * w, d[col_], width=w - 0.02, color=col, label=lab)
        ax.set_xticks(np.arange(len(folds)))
        ax.set_xticklabels([short.get(f, f) for f in folds], fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, out, 'fig_loso')


def fig_weight_sensitivity(tables, out):
    df = pd.read_csv(tables / 'weight_sensitivity_summary.csv')
    g = df[df.kind == 'grid']
    piv = g.pivot_table(index='w_p', columns='w_d', values='aurc')
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    vmin = np.nanmin(piv.values)
    vmax = min(np.nanmax(piv.values), vmin * 1.6)
    im = ax.imshow(piv.values, origin='lower', cmap='Blues_r', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f'{c:.1f}' for c in piv.columns], fontsize=7)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f'{c:.1f}' for c in piv.index], fontsize=7)
    ax.set_xlabel('$w_d$ (disablement weight)')
    ax.set_ylabel('$w_p$ (probability weight)')
    ax.grid(False)
    ref = np.where((piv.index.values.round(2) == 1.0))[0][0], np.where(piv.columns.values.round(2) == 0.0)[0][0]
    ax.scatter(
        [ref[1]], [ref[0]], marker='s', s=60, facecolor='none', edgecolor=PAL[7], lw=1.5, label='(1, 0, 0), as used'
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label('AURC (lower is better; clipped)')
    ax.legend(fontsize=7.5, loc='upper right')
    _save(fig, out, 'fig_weight_sensitivity')


def fig_missingness(tables, out):
    df = pd.read_csv(tables / 'missingness_patterns_summary.csv')
    d = df[df.family == 'MCAR']
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    order = [
        ('PICL', 'measured gases only', 'PICL, measured gases only', PAL[0], '-', 'o'),
        ('PICL', 'kNN (k=5)', 'PICL, kNN completion', PAL[0], '--', 's'),
        ('PICL', 'SCM (label-blind)', 'PICL, SCM completion', PAL[0], ':', '^'),
        ('Random Forest', 'measured gases only', 'RF, measured gases only', PAL[1], '-', 'o'),
        ('Random Forest', 'kNN (k=5)', 'RF, kNN completion', PAL[1], '--', 's'),
        ('Random Forest', 'MICE', 'RF, MICE', PAL[1], ':', '^'),
        ('XGBoost native missing', 'native (NaN)', 'XGBoost, native', PAL[2], '-.', 'x'),
    ]
    for ds_, comp, lab, col, ls_, mk in order:
        x = d[(d.downstream == ds_) & (d.completion == comp)].sort_values('rate')
        axes[0].plot(x.rate * 100, x.acc_full, color=col, ls=ls_, marker=mk, ms=3.5, lw=1.4, label=lab)
    axes[0].set_xlabel('Gas entries removed at random (%)')
    axes[0].set_ylabel('Accuracy (full coverage)')
    x = d[(d.downstream == 'PICL') & (d.completion == 'measured gases only')].sort_values('rate')
    axes[1].plot(x.rate * 100, x.coverage, color=PAL[0], marker='o', ms=3.5, lw=1.5, label='realised coverage')
    axes[1].plot(
        x.rate * 100, x.accepted_acc, color=PAL[0], ls='--', marker='s', ms=3.5, lw=1.5, label='accepted accuracy'
    )
    axes[1].set_xlabel('Gas entries removed at random (%)')
    axes[1].set_ylabel('Proportion')
    axes[0].set_title('(a) Full-coverage accuracy', fontsize=8.5, color=TXT2)
    axes[1].set_title('(b) PICL deferral, measured gases only', fontsize=8.5, color=TXT2)
    axes[0].set_ylim(0.1, 0.72)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.8, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    axes[1].legend(fontsize=7, loc='upper right', frameon=False)
    fig.tight_layout(w_pad=1.5, rect=(0, 0.12, 1, 1))
    _save(fig, out, 'fig_missingness')


def fig_intervention_heatmap(tables, out):
    df = pd.read_csv(tables / 'intervention_effects_summary.csv')
    piv = df.pivot(index='fault', columns='gas', values='model_effect').loc[
        FAULTS, ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
    ]
    hard = df.pivot(index='fault', columns='gas', values='hard_edge').loc[FAULTS, ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']]
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    im = ax.imshow(piv.values, cmap='Blues', aspect='auto', vmin=0)
    ax.set_xticks(range(5))
    ax.set_xticklabels(['H$_2$', 'CH$_4$', 'C$_2$H$_2$', 'C$_2$H$_4$', 'C$_2$H$_6$'])
    ax.set_yticks(range(6))
    ax.set_yticklabels(FAULTS)
    ax.grid(False)
    for i in range(6):
        for j in range(5):
            v = piv.values[i, j]
            ax.text(
                j,
                i,
                f'{v:.2f}',
                ha='center',
                va='center',
                fontsize=7,
                color='white' if v > 0.6 * np.nanmax(piv.values) else TXT,
            )
            if hard.values[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=PAL[1], lw=1.6))
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label('Effect of do($f_k$=1) on gas (z units)')
    ax.set_title('Orange frame: hard (IEC) edge', fontsize=8, color=TXT2)
    _save(fig, out, 'fig_intervention_heatmap')


def fig_reliability(tables, out):
    """Reliability diagram of PICL's classifier probabilities before and after temperature scaling (pooled over ten splits)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _common import load_scores

    seeds = list(range(52, 62))
    raw_c, raw_ok, cal_c, cal_ok = [], [], [], []
    for sd in seeds:
        try:
            te = load_scores(sd, 'test')
        except Exception:
            continue
        y = te['label'].to_numpy()
        Praw = te[[f'picl_Phraw{k}' for k in range(6)]].to_numpy()
        Pcal = te[[f'picl_P{k}' for k in range(6)]].to_numpy()
        raw_c.append(Praw.max(1))
        raw_ok.append(Praw.argmax(1) == y)
        cal_c.append(Pcal.max(1))
        cal_ok.append(Pcal.argmax(1) == y)
    raw_c, raw_ok, cal_c, cal_ok = map(np.concatenate, (raw_c, raw_ok, cal_c, cal_ok))
    edges = np.linspace(0, 1, 16)

    def bins(c, ok):
        xs, ys, ns = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (c > lo) & (c <= hi)
            if m.sum() >= 5:
                xs.append(c[m].mean())
                ys.append(ok[m].mean())
                ns.append(m.sum())
        ece = sum(n * abs(x - y) for x, y, n in zip(xs, ys, ns)) / len(c)
        return np.array(xs), np.array(ys), np.array(ns), ece

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for ax, (c, ok), title, col in zip(
        axes,
        ((raw_c, raw_ok), (cal_c, cal_ok)),
        ('Classifier head before calibration', 'PICL after fusion and calibration'),
        (PAL[1], PAL[0]),
    ):
        x, y, n, ece = bins(c, ok)
        ax.plot([0, 1], [0, 1], color=TXT2, lw=0.8, ls='--')
        ax.plot(x, y, color=col, marker='o', ms=4, lw=1.6)
        ax.set_title(title, fontsize=9)
        ax.text(
            0.04,
            0.93,
            f'ECE = {ece:.3f}\n{len(c)} test records, 10 splits',
            transform=ax.transAxes,
            fontsize=7.5,
            va='top',
            color=TXT2,
        )
        ax.set_xlabel('Predicted probability')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel('Observed accuracy')
    _save(fig, out, 'fig_reliability')


def fig_graph_transition(tables, out):
    """Bipartite fault->gas graph: hard edges, and discoverable edges kept / dropped / added in Stage 2 (majority over splits)."""
    import matplotlib.patches as mpatches

    ed = pd.read_csv(tables / 'graph_transition_edges.csv')
    n_seeds = ed.seed.nunique()
    gases = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
    glab = {'H2': 'H$_2$', 'CH4': 'CH$_4$', 'C2H2': 'C$_2$H$_2$', 'C2H4': 'C$_2$H$_4$', 'C2H6': 'C$_2$H$_6$'}
    hard = [
        ('PD', 'H2'),
        ('D1', 'C2H2'),
        ('D1', 'H2'),
        ('D2', 'C2H2'),
        ('D2', 'H2'),
        ('T1', 'CH4'),
        ('T1', 'C2H6'),
        ('T2', 'C2H4'),
        ('T2', 'C2H6'),
        ('T3', 'C2H4'),
        ('T3', 'C2H6'),
    ]
    fg = ed[ed.src.isin(FAULTS) & ed.tgt.isin(gases)]
    cnt = fg.groupby(['src', 'tgt', 'status']).size().unstack(fill_value=0)
    YF, YG, R = 2.4, 0.0, 0.30
    fx = {f: i * 1.6 for i, f in enumerate(FAULTS)}
    gx = {g: 0.8 + i * 1.6 for i, g in enumerate(gases)}
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    style = {
        'kept': (PAL[0], '-', 1.6),
        'dropped in Stage 2': (PAL[7], ':', 1.3),
        'added in Stage 2': (PAL[5], '--', 1.4),
    }
    counts = {'kept': 0, 'dropped in Stage 2': 0, 'added in Stage 2': 0}
    for (src, tgt), row in cnt.iterrows():
        if row.sum() < n_seeds / 2 or (src, tgt) in hard:
            continue
        st = row.idxmax()
        counts[st] += 1
        col, ls, lw = style[st]
        ax.annotate(
            '',
            xy=(gx[tgt], YG + R),
            xytext=(fx[src], YF - R),
            arrowprops=dict(arrowstyle='-|>', color=col, ls=ls, lw=lw, shrinkA=0, shrinkB=0, alpha=0.9),
        )
    for src, tgt in hard:
        ax.annotate(
            '',
            xy=(gx[tgt], YG + R),
            xytext=(fx[src], YF - R),
            arrowprops=dict(arrowstyle='-|>', color=TXT, lw=1.6, shrinkA=0, shrinkB=0),
        )
    for f, x in fx.items():
        ax.add_patch(plt.Circle((x, YF), R, color='#fbe0c8', ec=PAL[1], lw=0.8, zorder=3))
        ax.text(x, YF, f, ha='center', va='center', fontsize=8, zorder=4)
    for g, x in gx.items():
        ax.add_patch(plt.Circle((x, YG), R, color='#d7e6fa', ec=PAL[0], lw=0.8, zorder=3))
        ax.text(x, YG, glab[g], ha='center', va='center', fontsize=8, zorder=4)
    ax.set_xlim(-1.3, 8.8)
    ax.set_ylim(-0.5, 2.9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.text(-1.25, YF, 'Faults', fontsize=8, color=TXT2, va='center')
    ax.text(-1.25, YG, 'Gases', fontsize=8, color=TXT2, va='center')
    handles = [
        mpatches.Patch(color=TXT, label='$G_{\\mathrm{hard}}$, fixed (11)'),
        mpatches.Patch(color=PAL[0], label=f"kept in Stage 2 ({counts['kept']})"),
        mpatches.Patch(color=PAL[7], label=f"dropped in Stage 2 ({counts['dropped in Stage 2']})"),
        mpatches.Patch(color=PAL[5], label=f"added in Stage 2 ({counts['added in Stage 2']})"),
    ]
    ax.legend(handles=handles, fontsize=7, loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    _save(fig, out, 'fig_graph_transition')


def fig_misspecification_sweep(tables, out):
    df = pd.read_csv(tables / 'misspecification_test.csv')
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.axvspan(1.7, 1.8, color=GRID, lw=0, zorder=0)
    ax.axhline(0, color=TXT2, lw=0.8)
    for key, lab, col, mk in ((0.0, 'Gaussian noise', PAL[0], 'o'), (3.0, 'Student-$t$ noise ($\\nu=3$)', PAL[1], 's')):
        g = df[df.tail_df == key].groupby('cov_ratio').gain_pct.agg(['mean', 'std', 'count']).reset_index()
        half = 1.96 * g['std'] / np.sqrt(g['count'])
        ax.errorbar(
            g.cov_ratio, g['mean'], yerr=half, color=col, marker=mk, ms=4, lw=1.5, capsize=2.5, elinewidth=1, label=lab
        )
    ax.annotate(
        'sign change\n1.7$-$1.8',
        xy=(1.75, 0),
        xytext=(1.95, -11),
        fontsize=7.5,
        color=TXT2,
        arrowprops=dict(arrowstyle='-', color=TXT2, lw=0.7),
    )
    ax.set_xlabel('Ratio of class-conditional noise scales')
    ax.set_ylabel('AURC gain over confidence (%)')
    ax.legend(fontsize=7.5, loc='lower right')
    _save(fig, out, 'fig_misspecification_sweep')


def fig_class_gas_distributions(tables, out):
    """Distribution of each gas by fault class (log10(1 + ppm))."""
    d = pd.read_csv(Path('data/dga_provenance.csv')).query(
        "group == 'single_fault' and label_confirmed and fault_record"
    )
    gases = [('H2', 'H$_2$'), ('CH4', 'CH$_4$'), ('C2H2', 'C$_2$H$_2$'), ('C2H4', 'C$_2$H$_4$'), ('C2H6', 'C$_2$H$_6$')]
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.25), sharey=True)
    for ax, (g, lab) in zip(axes, gases):
        data = [np.log10(1 + d.loc[d.fault_type == f, g].dropna().to_numpy()) for f in FAULTS]
        bp = ax.boxplot(
            data,
            widths=0.6,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker='.', markersize=2, markerfacecolor=TXT2, markeredgecolor='none'),
            medianprops=dict(color=TXT, lw=1.2),
            whiskerprops=dict(color=TXT2, lw=0.8),
            capprops=dict(color=TXT2, lw=0.8),
            boxprops=dict(lw=0.6, edgecolor=TXT2),
        )
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(PAL[0] if FAULTS[i] in ('PD', 'D1', 'D2') else PAL[1])
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, 7))
        ax.set_xticklabels(FAULTS, fontsize=7, rotation=0)
        ax.set_title(lab, fontsize=9)
        ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('log$_{10}$(1 + concentration, $\\mu$L/L)')
    fig.tight_layout(w_pad=0.4)
    _save(fig, out, 'fig_class_gas_distributions')


def main(tables, out):
    out.mkdir(parents=True, exist_ok=True)
    for fn in (
        fig_risk_coverage,
        fig_matched_coverage,
        fig_per_class,
        fig_loso,
        fig_weight_sensitivity,
        fig_missingness,
        fig_intervention_heatmap,
        fig_reliability,
        fig_graph_transition,
        fig_misspecification_sweep,
        fig_class_gas_distributions,
    ):
        try:
            fn(tables, out)
            print('ok', fn.__name__)
        except Exception as e:
            print('FAILED', fn.__name__, repr(e))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='tables')
    ap.add_argument('--out', default='figures')
    a = ap.parse_args()
    main(Path(a.tables), Path(a.out))
