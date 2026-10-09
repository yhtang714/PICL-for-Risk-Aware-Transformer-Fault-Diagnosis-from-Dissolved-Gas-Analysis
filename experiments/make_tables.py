"""LaTeX tables of the paper from the CSV files in tables/.

python experiments/make_tables.py --tables tables --out tex_tables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import aurc as _aurc, accepted_acc_at_coverage as _aac, load_scores

FAULTS = ['PD', 'D1', 'D2', 'T1', 'T2', 'T3']
N_EFF = 383


def f(x, d=3):
    return '--' if (x is None or (isinstance(x, float) and np.isnan(x))) else f'{x:.{d}f}'


def pm(m, s, d=3):
    return f'{f(m, d)} $\\pm$ {f(s, d)}'


def sg(x, d=3):
    return '--' if (x is None or (isinstance(x, float) and np.isnan(x))) else f'{x:+.{d}f}'


def pv(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return '--'
    return '$<0.001$' if p < 0.001 else f'{p:.3f}'


def wrap(body, cols, caption, label, notes='', size='\\footnotesize', star=False):
    env = 'table*' if star else 'table'
    return (
        f"\\begin{{{env}}}[htbp]\\centering{size}\\setlength{{\\tabcolsep}}{{4pt}}\n\\caption{{{caption}}}\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{body}\\bottomrule\n\\end{{tabular}}\n{notes}\n\\end{{{env}}}\n"
    )


def note(text):
    return f'\\par\\vspace{{2pt}}\\parbox{{0.97\\textwidth}}{{\\scriptsize {text}}}'


def t_data(T, out, prov='data/dga_provenance.csv'):
    d = pd.read_csv(prov)
    s = d.query("group == 'single_fault' and label_confirmed and fault_record")
    basis = {
        'IEC TC 10': 'inspection of faulty equipment in service',
        'IEEE DataPort': 'published labels; records duplicating IEC TC 10, published cases or NCEPR removed',
        'Published cases': 'inspected in-service cases and laboratory cases reported in ppm',
        'NCEPR': 'case book: fault cause per transformer from inspection or repair; samples above attention values',
        'NE Grid': 'utility repair records; samples above attention values; records duplicating NCEPR removed',
        'Fujian': 'hanging-core or disassembly inspection; samples above attention values',
    }
    body = 'Source & Records & Transf. & PD & D1 & D2 & T1 & T2 & T3 & Label basis\\\\\n\\midrule\n'
    for src in ['IEC TC 10', 'IEEE DataPort', 'Published cases', 'NCEPR', 'NE Grid', 'Fujian']:
        x = s[s.source == src]
        cnt = [int((x.fault_type == k).sum()) for k in FAULTS]
        body += f"{src} & {len(x)} & {x.case_id.nunique()} & " + ' & '.join(map(str, cnt)) + f" & {basis[src]}\\\\\n"
    cnt = [int((s.fault_type == k).sum()) for k in FAULTS]
    body += '\\midrule\n' + f"Total & {len(s)} & {s.case_id.nunique()} & " + ' & '.join(map(str, cnt)) + ' & \\\\\n'
    weak = d.query("group == 'single_fault' and not label_confirmed and fault_record")
    extra = d[d.group.isin(['abnormal_no_fault', 'normal'])]
    cap = (
        'Development data with record-level provenance after cross-source de-duplication. Attention values follow DL/T~722: H$_2>150$, C$_2$H$_2>5$, total hydrocarbons $>150\\,\\mu$L/L. '
        f'Not in the development set: {len(weak)} Fujian utility records whose labels were assigned by the GB/T~7252 three-ratio code (weak-label control only, Section~\\ref{{sec:weak}}), '
        f'{len(extra)} records without a diagnosed fault (Section~\\ref{{sec:nonfault}}), and simultaneous-fault records, which are excluded from this study.'
    )
    (out / 'tab_data.tex').write_text(wrap(body, '@{}lrrrrrrrrp{5.4cm}@{}', cap, 'tab:data', '', '\\scriptsize'))


def t_klsens(T, out):
    a = pd.read_csv(T / 'kl_sensitivity_summary.csv')
    body = '$\\lambda_{\\mathrm{KL}}$ & Effective weight & Max.\\ posterior shift & Fault$\\to$gas edges & Gas$\\to$gas edges & Acc. & AURC & Acc.\\ at 85\\,\\%\\\\\n\\midrule\n'
    for _, r in a.sort_values('kl_weight', ascending=False).iterrows():
        used = ' (as used)' if abs(r.kl_weight - 200) < 1e-6 else ''
        body += (
            f"{r.kl_weight:g}{used} & {r.kl_weight / N_EFF:.2f} & {r.max_post_move_mean:.3f} & {pm(r.n_fault_gas_mean, r.n_fault_gas_std, 1)} & {pm(r.n_gas_gas_mean, r.n_gas_gas_std, 1)} & "
            f"{pm(r.acc_mean, r.acc_std)} & {pm(r.aurc_mean, r.aurc_std)} & {pm(r.acc85_mean, r.acc85_std)}\\\\\n"
        )
    n = note(
        f'Five evaluation splits. The effective weight is $\\lambda_{{\\mathrm{{KL}}}}/n_{{\\mathrm{{eff}}}}$ with $n_{{\\mathrm{{eff}}}}\\approx{N_EFF}$. The posterior shift is the largest change of a posterior inclusion probability from its prior mean over the 39 discoverable edges (19 fault$\\to$gas, 20 gas$\\to$gas). '
        'Edge counts are discoverable edges retained by Eq.~\\eqref{eq:final_graph}; the eleven hard edges are always present, so the fault$\\to$gas count equals the SHD to the IEC reference. Accuracy at 85\\,\\% is at matched coverage.'
    )
    (out / 'tab_klsens.tex').write_text(
        wrap(
            body,
            '@{}lccccccc@{}',
            'Sensitivity of the retained structure and of the diagnostic results to the strength of the edge prior.',
            'tab:klsens',
            n,
            '\\scriptsize',
        )
    )


def t_alignment(T, out):
    ka = pd.read_csv(T / 'knowledge_alignment.csv')
    agg = ka.groupby('method').agg(
        ERH=('ERH', 'mean'),
        SHD=('SHD', 'mean'),
        SHD_sd=('SHD', 'std'),
        PPA=('PPA', 'mean'),
        PPA_sd=('PPA', 'std'),
        n_edges=('n_edges', 'mean'),
    )
    body = 'Method & ERH & SHD & PPA & Edges\\\\\n\\midrule\n'
    for m in ('PICL', 'NOTEARS', 'DCDI*'):
        r = agg.loc[m]
        body += f"{m.replace('*', '')} & {f(r.ERH, 3)} & {pm(r.SHD, r.SHD_sd, 1)} & {pm(r.PPA, r.PPA_sd, 3)} & {f(r.n_edges, 1)}\\\\\n"
    (out / 'tab_alignment.tex').write_text(
        wrap(
            body,
            '@{}lcccc@{}',
            'Agreement of the learned graphs with the IEC~60599 reference over ten splits, all learners restricted to fault$\\to$gas edges outside the forbidden set (PICL may also add gas$\\to$gas edges, none of which is retained). Higher ERH and PPA and lower SHD are better; ERH is $1$ for PICL by construction.',
            'tab:alignment',
        )
    )


def t_cf(T, out):
    cf = pd.read_csv(T / 'counterfactual_faithfulness_summary.csv')

    def _tex(v):
        v = (
            str(v)
            .replace('Non-Cause Independence', 'Undocumented-Pair Response')
            .replace('non-causal pairs', 'undocumented pairs')
            .replace('on causal pairs', 'on documented pairs')
            .replace('+/-', '$\\pm$')
            .replace('do(f=1)', 'do$(f_k{=}1)$')
            .replace('argmax-Es', '$\\arg\\max_k \\tilde{E}_s$')
            .replace('|y - E[y|do(f=0)]|', '$|y-m(\\mathbf{0})|$')
            .replace('|do|', '$|$effect$|$')
        )
        return v[:-1] + '$\\times$' if v.endswith('x') else v

    body = 'Test & Quantity & PICL & Chance\\\\\n\\midrule\n'
    for _, r in cf.iterrows():
        body += f"{_tex(r['test'])} & {_tex(r['metric'])} & {_tex(r['PICL'])} & {_tex(r['Random Baseline'])}\\\\\n"
    t2 = pd.read_csv(T / 'counterfactual_test2_cause_removal.csv').set_index('fault')
    ratios = {k: t2.loc[k, 'ratio_hard_over_nonhard'] for k in FAULTS}
    n_pass = sum(v > 1 for v in ratios.values())
    n = note(
        'Test 2 per fault (mean displacement from the zero-fault reference on the hard-edge gases divided by that on the other gases): '
        + ', '.join(f'{k} {v:.2f}' for k, v in ratios.items())
        + (
            '; the ratio exceeds 1 for all six faults.'
            if n_pass == 6
            else f'; the ratio exceeds 1 for {n_pass} of the six faults.'
        )
    )
    (out / 'tab_cf.tex').write_text(
        wrap(
            body,
            '@{}lp{6.4cm}cc@{}',
            'Interventional tests of the fitted SCM (ten splits); Test~1 is imposed by the positivity constraint, Tests~2--4 are not.',
            'tab:cf',
            n,
        )
    )


def t_design(T, out):
    d = pd.read_csv(T / 'design' / 'picl_v2_explore_summary.csv').set_index('system')
    rows = [
        ('GB (depth 4) & gases, ratios, SCM features$^{a}$', 'GB(d4) gas+ratios+SCM (balanced) [current]'),
        ('XGBoost & gases, ratios', 'XGB gas+ratios (balanced)'),
        ('Random forest & gases, ratios', 'RF gas+ratios (balanced)'),
        ('Random forest & gases, ratios, SCM features$^{a}$', 'RF gas+ratios+SCM (balanced)'),
        ('Random forest & gases, ratios, evidence $\\log q$', 'RF gas+ratios+SCM-loglik (balanced)'),
    ]
    body = 'Head & Inputs & Fusion & Acc. & Macro-F1 & AURC & Acc.\\ at 85\\,\\%\\\\\n\\midrule\n'
    for lab, key in rows:
        for fus, tag in (('--', 'head'), ('Eq.~\\eqref{eq:fusion}', 'head+SCM evidence')):
            r = d.loc[f'{key} | {tag} | confidence']
            chosen = key.startswith('RF gas+ratios+SCM-loglik') and tag != 'head'
            cells = [
                f"{lab}" if tag == 'head' else ' & ',
                fus,
                pm(r.acc_mean, r.acc_std),
                f(r.macro_f1_mean),
                pm(r.aurc_mean, r.aurc_std),
                f(r.acc85_matched_mean),
            ]
            line = (cells[0] + ' & ' + ' & '.join(cells[1:])) if tag == 'head' else (' & & ' + ' & '.join(cells[1:]))
            if chosen:
                line = line.replace(' & & ', ' & \\textbf{(chosen)} & ', 1)
            body += line + '\\\\\n'
    r = d.loc['PICL-v2 (nested: head, SCM evidence, gate)']
    body += (
        '\\midrule\n'
        + f"\\multicolumn{{3}}{{@{{}}l}}{{Head, fusion and gate re-selected inside every split$^{{b}}$}} & {pm(r.acc_mean, r.acc_std)} & {f(r.macro_f1_mean)} & {pm(r.aurc_mean, r.aurc_std)} & {f(r.acc85_matched_mean)}\\\\\n"
    )
    n = note(
        'Design splits $42$--$51$ only; none of these numbers is used for a comparison in the rest of Section~6. Every head is class-balanced and temperature-scaled on the calibration split; deferral on the (fused) probability; accuracy at 85\\,\\% at matched coverage. '
        '$^{a}$SCM conditional mean of the fault block, per-class residual norms, $\\tilde{E}_d$ and $\\tilde{E}_s$ (the 39-feature input of the earlier design). '
        '$^{b}$Head chosen by grouped five-fold cross-validation inside the training split, evidence weight and gate (confidence, composite of Eq.~\\eqref{eq:channel_transform}, or a learned deferral model) chosen on the calibration split.'
    )
    (out / 'tab_design.tex').write_text(
        wrap(
            body,
            '@{}llccccc@{}',
            'Design study for the classifier read-out on the design splits (mean $\\pm$ s.d.\\ over ten splits).',
            'tab:design',
            n,
        )
    )


BASE = [
    ('Random forest', 'Random forest'),
    ('XGBoost', 'XGBoost'),
    ('SVM', 'SVM'),
    ('ANN', 'ANN'),
    ('Random forest (gases only)', 'Random forest, gases only'),
]


def t_main(T, out):
    ms = pd.read_csv(T / 'metrics_per_seed.csv')
    agg = ms.groupby('method').agg(['mean', 'std'])
    sig = pd.read_csv(T / 'significance_tests.csv')
    sig = sig[sig.metric == 'accuracy'].set_index('comparison')
    other = pd.read_csv(T / 'main_performance_table.csv').set_index('method')
    body = (
        'Method & Inputs & Accuracy & Balanced acc. & Macro-F1 & MCC & Brier & Coverage & $p$ vs PICL\\\\\n\\midrule\n'
    )
    a = agg.loc['PICL (accepted only)']
    body += f"\\textbf{{PICL, accepted records}} & G, R, E & {pm(a[('accuracy', 'mean')], a[('accuracy', 'std')])} & -- & -- & -- & -- & {f(a[('coverage', 'mean')])} & --\\\\\n"
    a = agg.loc['PICL (all samples)']
    body += (
        f"\\textbf{{PICL, all records}} & G, R, E & {pm(a[('accuracy', 'mean')], a[('accuracy', 'std')])} & {pm(a[('balanced_accuracy', 'mean')], a[('balanced_accuracy', 'std')])} & "
        f"{pm(a[('macro_f1', 'mean')], a[('macro_f1', 'std')])} & {pm(a[('mcc', 'mean')], a[('mcc', 'std')])} & {f(a[('brier_multiclass', 'mean')])} & 1.000 & --\\\\\n\\midrule\n"
    )
    for key, lab in BASE:
        a = agg.loc[key]
        p = sig.loc[f'PICL vs {key}', 'p_corrected']
        inp = 'G' if 'gases only' in key else 'G, R'
        body += (
            f"{lab} & {inp} & {pm(a[('accuracy', 'mean')], a[('accuracy', 'std')])} & {pm(a[('balanced_accuracy', 'mean')], a[('balanced_accuracy', 'std')])} & "
            f"{pm(a[('macro_f1', 'mean')], a[('macro_f1', 'std')])} & {pm(a[('mcc', 'mean')], a[('mcc', 'std')])} & {f(a[('brier_multiclass', 'mean')])} & 1.000 & {pv(p)}\\\\\n"
        )
    cbp = T / 'v6_catboost_eval_per_seed.csv'
    if cbp.exists():
        cb = pd.read_csv(cbp).set_index('seed')
        pa = ms[ms.method == 'PICL (all samples)'].set_index('seed').accuracy
        diff = (pa.loc[cb.index] - cb.accuracy).values
        from scipy import stats as _st

        J = len(diff)
        se = np.sqrt((1 / J + 127 / 383) * diff.var(ddof=1))
        pcb = float(2 * _st.t.sf(abs(diff.mean() / se), J - 1))
        body += (
            f"CatBoost & G, R & {pm(cb.accuracy.mean(), cb.accuracy.std())} & {pm(cb.balanced_accuracy.mean(), cb.balanced_accuracy.std())} & "
            f"{pm(cb.macro_f1.mean(), cb.macro_f1.std())} & {pm(cb.mcc.mean(), cb.mcc.std())} & {f(cb.brier_multiclass.mean())} & 1.000 & {pv(pcb)}\\\\\n"
        )
    p = T / 'modern_baselines.csv'
    if p.exists():
        md = (
            pd.read_csv(p)
            .groupby('method')
            .agg(
                acc=('accuracy', 'mean'),
                acc_sd=('accuracy', 'std'),
                bal=('balanced_accuracy', 'mean'),
                bal_sd=('balanced_accuracy', 'std'),
                f1=('macro_f1', 'mean'),
                f1_sd=('macro_f1', 'std'),
                mcc=('mcc', 'mean'),
                mcc_sd=('mcc', 'std'),
            )
            .reset_index()
            .sort_values('acc', ascending=False)
        )
        for _, r in md.iterrows():
            inp = 'G, R' if r.method == 'Diffusion+RF' else 'G'
            body += f"{r.method}$^{{\\dagger}}$ & {inp} & {pm(r.acc, r.acc_sd)} & {pm(r.bal, r.bal_sd)} & {pm(r.f1, r.f1_sd)} & {pm(r.mcc, r.mcc_sd)} & -- & 1.000 & --\\\\\n"
    for key, lab, inp in (
        ('GMM', 'GMM', 'G'),
        ('GNB', 'GNB', 'G'),
        ('NOTEARS [fault_to_gas_only]', 'NOTEARS', 'G'),
        ('DCDI* [fault_to_gas_only]', 'DCDI', 'G'),
        ('HillClimb [fault_to_gas_only]', 'HillClimb', 'G'),
    ):
        r = other.loc[key]
        body += f"{lab} & {inp} & {pm(r.acc_mean, r.acc_std)} & {pm(r.bal_acc_mean, r.bal_acc_std)} & {pm(r.macro_f1_mean, r.macro_f1_std)} & {pm(r.mcc_mean, r.mcc_std)} & -- & 1.000 & --\\\\\n"
    rb = pd.read_csv(T / 'rule_based_coverage_breakdown.csv')
    for _, r in rb.iterrows():
        body += (
            f"{r.method} & ratios & {f(r.acc_on_covered)} (covered) & -- & -- & -- & -- & {f(r.coverage)} & --\\\\\n"
        )
    msub = pd.read_csv(T / 'metrics_per_seed.csv')
    md_seeds = sorted(pd.read_csv(p).seed.unique()) if p.exists() else []
    ref = msub[msub.seed.isin(md_seeds) & (msub.method == 'PICL (all samples)')].accuracy
    n = note(
        'Ten evaluation splits, $n_{\\text{test}}\\approx128$. Inputs: G = five gases (PICL: measured gases only, no imputation; baselines: unmeasured gases completed label-blind), R = pairwise log-ratios, E = interventional evidence of the SCM. '
        'Brier: multiclass Brier score of the calibrated probabilities (lower is better). PICL on accepted records uses the threshold chosen on the calibration split for $85\\,\\%$ coverage. $p$ values: corrected resampled $t$-test on accuracy over the ten splits. '
        f'$^{{\\dagger}}$First five evaluation splits, on which PICL reaches {f(ref.mean())} $\\pm$ {f(ref.std())} accuracy. Rule-based methods are scored on the records they cover.'
    )
    (out / 'tab_main.tex').write_text(
        wrap(
            body,
            '@{}llcccccrc@{}',
            'Diagnostic performance at full coverage (mean $\\pm$ s.d.).',
            'tab:main',
            n,
            '\\scriptsize',
        )
    )


def t_perclass(T, out):
    pc = pd.read_csv(T / 'per_class_metrics.csv')
    body = 'Class & $n_{\\text{test}}$ & \\multicolumn{3}{c}{PICL} & \\multicolumn{3}{c}{Random forest} & \\multicolumn{3}{c}{XGBoost}\\\\\n'
    body += ' & & P & R & F1 & P & R & F1 & P & R & F1\\\\\n\\midrule\n'
    for k in FAULTS:
        cells = [k, f(pc[(pc.fault == k) & (pc.method == 'PICL (all samples)')].n.mean(), 1)]
        for m in ('PICL (all samples)', 'Random forest', 'XGBoost'):
            d = pc[(pc.fault == k) & (pc.method == m)]
            cells += [f(d.precision.mean()), f(d.recall.mean()), f(d.f1.mean())]
        body += ' & '.join(cells) + '\\\\\n'
    (out / 'tab_perclass.tex').write_text(
        wrap(
            body,
            '@{}lrccccccccc@{}',
            'Class-wise precision (P), recall (R) and F1 (mean over ten evaluation splits; baselines with the same gas and ratio inputs).',
            'tab:perclass',
        )
    )


def t_paired(T, out):
    b = pd.read_csv(T / 'paired_differences.csv')
    names = {'acc': 'accuracy', 'macro_f1': 'macro-F1', 'aurc': 'AURC', 'acc85': 'acc.\\ at 85\\,\\%'}
    comps = [
        ('PICL - random forest', 'Random forest'),
        ('PICL - XGBoost', 'XGBoost'),
        ('PICL - SVM', 'SVM'),
        ('PICL - ANN', 'ANN'),
        ('PICL - random forest (gases only)', 'Random forest, gases only'),
        ('PICL - PICL head without SCM evidence', 'PICL without evidence fusion'),
    ]
    rec = (
        pd.read_csv(T / 'v6_indist_transformer_tests.csv').set_index('vs')
        if (T / 'v6_indist_transformer_tests.csv').exists()
        else None
    )
    recmap = {'Random forest': 'Random forest', 'XGBoost': 'XGBoost', 'SVM': 'SVM', 'CatBoost': 'CatBoost'}
    cbp = T / 'v6_catboost_eval_per_seed.csv'
    if cbp.exists():
        cb = pd.read_csv(cbp).set_index('seed')
        ms = pd.read_csv(T / 'metrics_per_seed.csv')
        pa = ms[ms.method == 'PICL (all samples)'].set_index('seed')
        sc = {}
        for seed in cb.index:
            te = load_scores(seed, 'test')
            P = te[[f'picl_P{k}' for k in range(6)]].values
            y = te.label.values
            c = P.argmax(1) == y
            sc[seed] = dict(
                acc=c.mean(),
                macro_f1=pa.loc[seed, 'macro_f1'],
                aurc=_aurc(P.max(1), c),
                acc85=_aac(P.max(1), c, 0.85)['accepted_acc'],
            )
        from scipy import stats as _st

        extra = []
        for met, col in (('acc', 'accuracy'), ('macro_f1', 'macro_f1'), ('aurc', 'aurc'), ('acc85', 'acc85')):
            diff = np.array([sc[sd][met] - cb.loc[sd, col] for sd in cb.index])
            J = len(diff)
            se = np.sqrt((1 / J + 127 / 383) * diff.var(ddof=1))
            t = diff.mean() / se
            extra.append(
                dict(
                    comparison='PICL - CatBoost',
                    metric=met,
                    mean_diff=diff.mean(),
                    ci_corrected_lo=diff.mean() - _st.t.ppf(0.975, J - 1) * se,
                    ci_corrected_hi=diff.mean() + _st.t.ppf(0.975, J - 1) * se,
                    p_corrected=float(2 * _st.t.sf(abs(t), J - 1)),
                    p_ttest=float(_st.ttest_1samp(diff, 0).pvalue),
                    wins=int((diff > 0).sum()) if met != 'aurc' else int((diff < 0).sum()),
                    n_splits=J,
                    range_lo=diff.min(),
                    range_hi=diff.max(),
                )
            )
        b = pd.concat([b, pd.DataFrame(extra)], ignore_index=True)
        comps = comps[:1] + [('PICL - CatBoost', 'CatBoost')] + comps[1:]
    body = (
        'PICL minus & Metric & Mean & \\multicolumn{2}{c}{Split-level corrected test} & PICL better & \\multicolumn{1}{c}{Transformer-level bootstrap}\\\\\n'
        '\\cmidrule(lr){4-5}\\cmidrule(lr){7-7}\n & & & 95\\,\\% CI & $p$ & (splits) & 95\\,\\% CI\\\\\n\\midrule\n'
    )
    for comp, lab in comps:
        first = True
        for met in ('acc', 'macro_f1', 'aurc', 'acc85'):
            r = b[(b.comparison == comp) & (b.metric == met)]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            rc = '--'
            if rec is not None and lab in recmap and recmap[lab] in rec.index and met in ('acc', 'aurc'):
                rr = rec.loc[recmap[lab]]
                lo, hi = (rr.acc_ci_lo, rr.acc_ci_hi) if met == 'acc' else (rr.aurc_ci_lo, rr.aurc_ci_hi)
                star = '$^{*}$' if (lo > 0 or hi < 0) else ''
                rc = f'[{lo:+.3f}, {hi:+.3f}]{star}'
            body += (
                f"{lab if first else ''} & {names[met]} & {r.mean_diff:+.3f} & [{r.ci_corrected_lo:+.3f}, {r.ci_corrected_hi:+.3f}] & {pv(r.p_corrected)} & "
                f"{int(r.wins)}/{int(r.n_splits)} & {rc}\\\\\n"
            )
            first = False
        if comp != comps[-1][0]:
            body += '\\addlinespace[1pt]\n'
    n = note(
        'Ten evaluation splits; differences are PICL minus the method named (for AURC, negative favours PICL). The random forest receives exactly the inputs of the PICL head except the evidence, so its row measures the effect of the SCM evidence as a whole; the row without evidence fusion keeps the evidence as head input. '
        'Split-level test: corrected resampled $t$-test of \\citet{nadeau2003inference} over the ten splits. Transformer-level test: cluster bootstrap over the 330 transformers or cases whose 604 records are test records in at least one evaluation split (1\\,277 pooled out-of-fold predictions; each resampled transformer brings all its records and their predictions; 4\\,000 resamples); $^{*}$ interval excludes zero. '
        'Accuracy at 85\\,\\% is at matched coverage.'
    )
    (out / 'tab_paired.tex').write_text(
        wrap(
            body,
            '@{}llccccc@{}',
            'Paired differences between PICL and the other methods.',
            'tab:paired',
            n,
            '\\scriptsize',
        )
    )


SEL = [
    ('picl', '\\textbf{PICL}'),
    ('picl_head', 'PICL without evidence fusion'),
    ('picl_composite', 'PICL with $\\tilde{E}_d,\\tilde{E}_s$ in the gate'),
    ('random_forest', 'Random forest'),
    ('xgboost', 'XGBoost'),
    ('svm', 'SVM'),
    ('ann', 'ANN'),
    ('random_forest_gases_only', 'Random forest, gases only'),
    ('random_forest+causal', 'Random forest with $\\tilde{E}_d,\\tilde{E}_s$'),
    ('xgboost+causal', 'XGBoost with $\\tilde{E}_d,\\tilde{E}_s$'),
]


def t_selective(T, out):
    a = pd.read_csv(T / 'aurc_all_methods.csv').set_index('key')
    mc = pd.read_csv(T / 'matched_coverage_summary.csv')
    body = (
        'Score & AURC & \\multicolumn{4}{c}{Accepted accuracy, matched coverage} & \\multicolumn{2}{c}{Transferred 85\\,\\%} & Errors\\\\\n'
        ' & & 60\\,\\% & 70\\,\\% & 85\\,\\% & 95\\,\\% & coverage & accuracy & at 85\\,\\%\\\\\n\\midrule\n'
    )
    for key, name in SEL:
        r = a.loc[key]
        cells = [name, pm(r.aurc, r.aurc_sd)]
        for c in (0.60, 0.70, 0.85, 0.95):
            x = mc[(mc.key == key) & (np.isclose(mc.target_coverage, c))]
            cells.append(f(x.matched_acc.iloc[0]))
        x = mc[(mc.key == key) & (np.isclose(mc.target_coverage, 0.85))].iloc[0]
        cells += [f(x.transferred_coverage), f(x.transferred_acc), f(x.matched_errors, 1)]
        body += ' & '.join(cells) + '\\\\\n'
        if key in ('picl_composite', 'random_forest_gases_only'):
            body += '\\addlinespace[1pt]\n'
    n = note(
        'Ten evaluation splits, identical test records and inputs for every score. Matched coverage accepts the same fraction of test records for every score; transferred coverage applies the threshold chosen on the calibration split. '
        'Errors are mean accepted misdiagnoses per split at 85\\,\\% matched coverage (about 109 accepted records). Rows with $\\tilde{E}_d,\\tilde{E}_s$ add the intervention scores to the respective probability as in Eq.~\\eqref{eq:channel_transform} with weights $(0.4,0.3,0.3)$.'
    )
    (out / 'tab_selective.tex').write_text(
        wrap(body, '@{}lcccccccc@{}', 'Selective diagnosis on identical inputs.', 'tab:selective', n, '\\scriptsize')
    )


def t_gate(T, out):
    t = pd.read_csv(T / 'transform_alternatives_summary.csv')
    tt = pd.read_csv(T / 'transform_alternatives_tests.csv').set_index('variant')
    w = pd.read_csv(T / 'weight_sensitivity_summary.csv')
    g = w[w.kind == 'grid']
    conf = g[g.w_p == 1.0].iloc[0]
    ref = g[(np.isclose(g.w_p, 0.4)) & (np.isclose(g.w_d, 0.3))].iloc[0]
    best = g.sort_values('aurc').iloc[0]
    eq = w[w.kind == 'equal'].iloc[0]
    lr = w[w.kind.str.startswith('learned')].iloc[0]
    beat = int((g.aurc < conf.aurc - 1e-9).sum())
    within = int((g.aurc <= conf.aurc + conf.aurc_sd).sum())
    names = {
        'eq36': 'fixed sample-wise maps of Eq.~\\eqref{eq:channel_transform}',
        'isotonic': 'isotonic per channel',
        'learned': 'logistic combiner',
        'zscore_logistic': '$z$-score + logistic',
        'rank': 'calibration-set rank',
        'minmax': 'calibration-set min--max',
        'raw': 'raw channels (no map)',
        'clipped': 'clipped channels',
        'confidence': 'fused probability only (as used)',
    }
    body = 'Deferral score & AURC & AURC reduction vs.\\ fused probability & Acc.\\ at 85\\,\\%\\\\\n\\midrule\n'
    body += '\\emph{Normalisation of $\\tilde{E}_d,\\tilde{E}_s$, weights $(0.4,0.3,0.3)$} & & & \\\\\n'
    for _, r in t.sort_values('aurc').iterrows():
        if r.variant == 'confidence':
            gain = '--'
        elif r.variant in tt.index:
            gain = f"{tt.loc[r.variant, 'gain_vs_confidence']:+.1f}\\,\\% ($p={tt.loc[r.variant, 'p']:.2f}$)"
        else:
            gain = '--'
        body += (
            f"\\quad {names.get(r.variant, r.variant)} & {pm(r.aurc, r.aurc_sd)} & {gain} & {f(r.matched_acc85)}\\\\\n"
        )
    body += '\\addlinespace[2pt]\\emph{Weights $(w_p,w_d,w_s)$, maps of Eq.~\\eqref{eq:channel_transform}} & & & \\\\\n'
    for name, r in (
        ('(1, 0, 0), fused probability only (as used)', conf),
        (f'({best.w_p:.1f}, {best.w_d:.1f}, {best.w_s:.1f}), best grid point', best),
        ('(0.4, 0.3, 0.3)', ref),
        ('(1/3, 1/3, 1/3)', eq),
        ('logistic stacker (calibration split)', lr),
    ):
        body += f"\\quad {name} & {pm(r.aurc, r.aurc_sd)} & -- & {f(r.matched_acc85)}\\\\\n"
    n = note(
        f'Ten evaluation splits; every variant is fitted on the calibration split only and the AURC reduction is relative to the fused probability (positive is better; $p$ uncorrected paired $t$). '
        f'Of the 66 grid points of the weight simplex (step 0.1), {beat} have a lower mean AURC than the fused probability alone and {within} lie within one split-to-split standard deviation of it.'
    )
    (out / 'tab_gate.tex').write_text(
        wrap(
            body,
            '@{}lccc@{}',
            'Adding the intervention scores to the deferral score: normalisation and weights.',
            'tab:gate',
            n,
        )
    )
    return dict(beat=beat, within=within, best=best, conf=conf, ref=ref, lr=lr)


def t_residuals(T, out):
    r = pd.read_csv(T / 'residual_diagnostics.csv')
    g = (
        r[r.is_gas]
        .groupby('variable')
        .agg(
            skew=('skew', 'mean'),
            kurt=('excess_kurtosis', 'mean'),
            sw=('shapiro_p', 'mean'),
            bp=('breusch_pagan_p', 'mean'),
        )
    )
    lab = {'H2': 'H$_2$', 'CH4': 'CH$_4$', 'C2H2': 'C$_2$H$_2$', 'C2H4': 'C$_2$H$_4$', 'C2H6': 'C$_2$H$_6$'}
    body = 'Gas & Skewness & Excess kurtosis & Shapiro--Wilk $p$ & Breusch--Pagan $p$\\\\\n\\midrule\n'
    for gname in ('H2', 'CH4', 'C2H2', 'C2H4', 'C2H6'):
        x = g.loc[gname]
        body += f"{lab[gname]} & {x['skew']:+.2f} & {x['kurt']:.2f} & {x['sw']:.3f} & {x['bp']:.3f}\\\\\n"
    (out / 'tab_residuals.tex').write_text(
        wrap(
            body,
            '@{}lcccc@{}',
            'Residual diagnostics of the fitted linear-Gaussian SCM (mean over ten evaluation splits).',
            'tab:residuals',
        )
    )


def t_scmvariants(T, out):
    from scipy import stats as _st

    df = pd.read_csv(T / 'scm_variants.csv')
    piv = df.pivot(index='seed', columns='variant', values='downstream_acc')
    rm = df.pivot(index='seed', columns='variant', values='test_rmse')
    ref = piv['linear-log1p_z']
    names = {
        'linear-log1p_z': 'linear-Gaussian, $\\log(1+x)$ $z$-scores (as used)',
        'mlp-log1p_z': 'MLP additive-noise, $\\log(1+x)$ $z$-scores',
        'linear-clr': 'linear-Gaussian, centred log-ratio$^{\\ddagger}$',
        'mlp-clr': 'MLP additive-noise, centred log-ratio',
    }
    body = 'Mechanism / feature space & Held-out residual RMSE & Downstream accuracy & $\\Delta$ vs.\\ as used ($p$)\\\\\n\\midrule\n'
    for v in ('linear-log1p_z', 'mlp-log1p_z', 'linear-clr', 'mlp-clr'):
        d = piv[v] - ref
        pval = _st.ttest_rel(piv[v], ref).pvalue if v != 'linear-log1p_z' else np.nan
        delta = '--' if v == 'linear-log1p_z' else f"{d.mean():+.3f} ($p={pval:.2f}$)"
        body += f"{names[v]} & {rm[v].mean():.3f} & {pm(piv[v].mean(), piv[v].std())} & {delta}\\\\\n"
    n = note(
        'Five evaluation splits; every mechanism is fitted on the same structure (fault indicators and the remaining gases as parents) and scored by a common class-balanced random forest on the gases plus the per-class residual norms; $p$ uncorrected paired $t$. RMSE is comparable only within a feature space. '
        '$^{\\ddagger}$Centred log-ratio coordinates sum to zero, so a linear mechanism conditioned on the other four coordinates reproduces each gas exactly (RMSE 0) and its residual features carry no information; the row is kept for completeness.'
    )
    (out / 'tab_scmvariants.tex').write_text(
        wrap(
            body,
            '@{}lccc@{}',
            'Sensitivity of the downstream diagnosis to the SCM mechanism family and feature space.',
            'tab:scmvariants',
            n,
        )
    )


LOSO_M = [
    ('PICL unseen', 'PICL, unseen-source read-out'),
    ('PICL', 'PICL, in-distribution read-out'),
    ('Random Forest', 'Random forest'),
    ('XGBoost', 'XGBoost'),
    ('SVM', 'SVM'),
    ('ANN', 'ANN'),
    ('Random Forest (gases only)', 'Random forest, gases only'),
]
LOSO_F = [
    ('IEC TC 10', 'IEC TC 10'),
    ('IEEE DataPort', 'IEEE DataPort'),
    ('Published cases', 'Published cases'),
    ('NCEPR', 'NCEPR'),
    ('NE Grid+Fujian', 'NE Grid + Fujian'),
]


def _loso_with_v4(T):
    a = pd.read_csv(T / 'loso_summary.csv')
    pf = pd.read_csv(T / 'loso_per_fold.csv')
    v4 = T / 'v4_loso_per_fold.csv'
    if v4.exists():
        u = pd.read_csv(v4)
        u = u[u.method == 'PICL, unseen-source read-out'].copy()
        u['method'] = 'PICL unseen'
        pf = pd.concat([pf, u[['seed', 'held_out', 'n_test', 'method', 'acc', 'macro_f1', 'aurc']]], ignore_index=True)
        ua = (
            u.groupby('held_out')
            .agg(n_test=('n_test', 'first'), acc=('acc', 'mean'), macro_f1=('macro_f1', 'mean'), aurc=('aurc', 'mean'))
            .reset_index()
        )
        ua['method'] = 'PICL unseen'
        a = pd.concat([a, ua], ignore_index=True)
    return a, pf


def t_loso(T, out):
    a, pf = _loso_with_v4(T)
    micro = (
        pf.groupby(['seed', 'method'])
        .apply(
            lambda d: pd.Series(
                {
                    'acc': np.average(d.acc, weights=d.n_test),
                    'aurc': np.average(d.aurc, weights=d.n_test),
                    'f1': np.average(d.macro_f1, weights=d.n_test),
                }
            ),
            include_groups=False,
        )
        .groupby('method')
        .mean()
    )
    hdr = ' & '.join(n for _, n in LOSO_F)
    ns = ' & '.join(f"($n$={int(a[a.held_out == k].n_test.iloc[0])})" for k, _ in LOSO_F)
    body = f'Method & {hdr} & Micro-average\\\\\n & {ns} & \\\\\n\\midrule\n'
    for key, name in LOSO_M:
        vals = [f(a[(a.method == key) & (a.held_out == fk)]['acc'].iloc[0]) for fk, _ in LOSO_F]
        body += f"{name} & " + ' & '.join(vals) + f" & {f(micro.loc[key, 'acc'])}\\\\\n"
    lo = a[a.method == 'PICL'].transferred_coverage
    n = note(
        f'Five seeds per held-out source. The held-out source is unseen during training, calibration and threshold selection; the transferred 85\\,\\% threshold of the in-distribution read-out realises {lo.min():.2f}--{lo.max():.2f} coverage on it. Unseen-source read-out: Section~\\ref{{sec:unseen}}. Micro-averages weight the folds by their size; macro-F1 per source in Figure~\\ref{{fig:loso}}; micro-averaged macro-F1 and AURC in Table~\\ref{{tab:losofam}}.'
    )
    (out / 'tab_loso.tex').write_text(
        wrap(
            body,
            '@{}lcccccc@{}',
            'Leave-one-source-out transfer: accuracy on each held-out source.',
            'tab:loso',
            n,
            '\\scriptsize',
        )
    )


def t_weak(T, out):
    a = pd.read_csv(T / 'weak_label_control_summary.csv')
    t = pd.read_csv(T / 'weak_label_control_tests.csv')
    body = 'Method & Training records & Acc. & Balanced acc. & Macro-F1 & AURC & Agreement with three-ratio labels\\\\\n\\midrule\n'
    for m, lab in (('PICL', 'PICL'), ('Random Forest', 'Random forest'), ('XGBoost', 'XGBoost')):
        for arm in ('confirmed only', 'confirmed + rule-labelled'):
            r = a[(a.method == m) & (a.arm == arm)].iloc[0]
            body += f"{lab if arm == 'confirmed only' else ''} & {arm} ({int(round(r.n_train_mean))}) & {f(r.acc_mean)} & {f(r.bal_acc_mean)} & {f(r.macro_f1_mean)} & {f(r.aurc_mean)} & {f(r.three_ratio_agreement_mean)}\\\\\n"
    parts = []
    for m, lab in (('PICL', 'PICL'), ('Random Forest', 'random forest'), ('XGBoost', 'XGBoost')):
        for met, ml in (('acc', 'accuracy'), ('bal_acc', 'balanced accuracy'), ('macro_f1', 'macro-F1')):
            r = t[(t.method == m) & (t.metric == met)].iloc[0]
            parts.append(f"{lab} {ml} $\\Delta={r.delta:+.3f}$ ($p={r.p_corrected:.2f}$)")
    n = note(
        'Ten evaluation splits; rule-labelled records enter the training split only; $p$ corrected resampled $t$-test. '
        + '; '.join(parts)
        + '.'
    )
    (out / 'tab_weak.tex').write_text(
        wrap(
            body,
            '@{}llccccc@{}',
            'Weak-label control: adding the 412 rule-labelled utility records to training.',
            'tab:weak',
            n,
        )
    )


def t_missing(T, out):
    m = pd.read_csv(T / 'missingness_patterns_summary.csv')
    pats = [
        ('none', 'complete records', None),
        ('mcar_10', 'MCAR 10\\,\\%', None),
        ('mcar_30', 'MCAR 30\\,\\%', None),
        ('mcar_50', 'MCAR 50\\,\\%', None),
        ('single_C2H2', 'C$_2$H$_2$ removed', None),
        ('single_H2', 'H$_2$ removed', None),
        ('pair_H2_C2H2', 'H$_2$, C$_2$H$_2$ removed', None),
        ('pair_C2H4_C2H6', 'C$_2$H$_4$, C$_2$H$_6$ removed', None),
        ('censor_20', 'below 20th pct., unmeasured', None),
        ('censor_20', 'below 20th pct., half limit', 'half detection limit'),
        ('natural (incomplete records)', 'naturally incomplete', None),
    ]
    cols = [
        ('PICL', 'measured gases only'),
        ('PICL', 'kNN (k=5)'),
        ('PICL', 'SCM (label-blind)'),
        ('Random Forest', 'measured gases only'),
        ('Random Forest', 'kNN (k=5)'),
        ('Random Forest', 'SCM (label-blind)'),
        ('Random Forest', 'MICE'),
        ('XGBoost native missing', 'native (NaN)'),
    ]
    body = (
        'Pattern & \\multicolumn{3}{c}{PICL} & \\multicolumn{4}{c}{Random forest} & XGBoost & PICL\\\\\n'
        ' & measured & $k$NN & SCM & measured & $k$NN & SCM & MICE & native & coverage\\\\\n\\midrule\n'
    )
    for key, name, alt in pats:
        x = m[m.pattern == key]
        if len(x) == 0:
            continue
        cells = [name]
        for ds, comp in cols:
            c = alt if (alt is not None and comp == 'measured gases only') else comp
            if alt is not None and comp != 'measured gases only':
                cells.append('--')
                continue
            y = x[(x.downstream == ds) & (x.completion == c)]
            cells.append(f(y.acc_full.iloc[0]) if len(y) else '--')
        p_m = x[(x.downstream == 'PICL') & (x.completion == (alt or 'measured gases only'))]
        cells.append(f(p_m.coverage.iloc[0], 2) if len(p_m) else '--')
        body += ' & '.join(cells) + '\\\\\n'
    nn = (
        m[m.pattern == 'natural (incomplete records)'].n.mean()
        if (m.pattern == 'natural (incomplete records)').any()
        else np.nan
    )
    n = note(
        'Full-coverage accuracy (ten evaluation splits; SCM and main read-out fixed, pattern read-outs fitted on the training records). Measured: no imputation (PICL as proposed; random forest trained on the measured gases and their ratios). '
        '$k$NN, SCM, MICE: completion of the unmeasured gases. Half limit: values below the detection limit reported at half the limit. '
        f'Coverage: realised coverage of PICL (measured). Naturally incomplete: {nn:.1f} test records per split. All patterns are reported in the accompanying code.'
    )
    (out / 'tab_missing.tex').write_text(
        wrap(
            body,
            '@{}lccccccccc@{}',
            'Diagnosis under controlled and natural missingness.',
            'tab:missing',
            n,
            '\\scriptsize',
        )
    )


def t_nonfault(T, out):
    n_ = pd.read_csv(T / 'nonfault_screening_summary.csv')
    r = n_[n_.group == 'all'].iloc[0]
    lab = pd.read_csv(T / 'nonfault_predicted_labels.csv')
    body = 'Quantity & Records without a diagnosed fault & Fault records (test)\\\\\n\\midrule\n'
    body += f"Accepted under the 85\\,\\% threshold (PICL) & {f(r.accept_rate_85_mean)} & {f(r.fault_records_accept_rate_85_mean)}\\\\\n"
    body += f"Mean maximum fused probability & {f(r.mean_max_prob_mean)} & {f(r.mean_max_prob_fault_mean)}\\\\\n"
    body += f"Mean disablement score $\\tilde E_d$ & {f(r.mean_Ed_mean)} & {f(r.mean_Ed_fault_mean)}\\\\\n"
    body += f"Fraction exceeding attention values & {f(r.exceeds_attention_frac_mean)} & 1.000\\\\\n"
    top = lab.sort_values('frac_predicted', ascending=False).iloc[0]
    n = note(
        f"{int(r.n_mean)} records (69 abnormal-gas utility records without a diagnosed fault, 2 normal samples), ten evaluation splits. The most frequent assigned label is {top.label} ({100 * top.frac_predicted:.0f}\\,\\%)."
    )
    (out / 'tab_nonfault.tex').write_text(
        wrap(
            body,
            '@{}lcc@{}',
            'Behaviour of the deferral rule on records without a diagnosed fault.',
            'tab:nonfault',
            n,
            '\\scriptsize',
        )
    )
    return r, top


def t_cost(T, out):
    c = pd.read_csv(T / 'matched_cost_summary.csv')
    rows = [
        ('picl', 'PICL'),
        ('picl_composite', 'PICL with $\\tilde{E}_d,\\tilde{E}_s$'),
        ('random_forest', 'Random forest'),
        ('xgboost', 'XGBoost'),
        ('svm', 'SVM'),
    ]
    body = 'Score & $r=1$ & $r=2$ & $r=5$ & $r=10$ & $r=20$ & $r=50$\\\\\n\\midrule\n'
    for key, name in rows:
        x = c[c.key == key].set_index('r')
        body += (
            name
            + ' & '
            + ' & '.join(f"{100 * x.loc[r, 'cost_reduction']:.1f}\\,\\%" for r in (1, 2, 5, 10, 20, 50))
            + '\\\\\n'
        )
    n = note(
        'Ten evaluation splits; reduction of $C=r\\,n_{\\text{err}}+n_{\\text{defer}}$ on the test split relative to answering every record, with the threshold chosen on the calibration split for each $r$ and each score.'
    )
    (out / 'tab_cost.tex').write_text(
        wrap(
            body,
            '@{}lcccccc@{}',
            'Cost-sensitive operation: expected-cost reduction versus no deferral.',
            'tab:cost',
            n,
        )
    )


ABL = [
    ('PICL (Full)', 'PICL (full)'),
    ('- SCM evidence (features and fusion)', '$-$ SCM evidence (features and fusion)'),
    ('- SCM evidence as head features (fusion kept)', '$-$ evidence as head input (fusion kept)'),
    ('- SCM evidence fusion (features kept)', '$-$ evidence fusion (input kept)'),
    ('- log-ratio features', '$-$ log-ratio inputs'),
    ('- G_data (only G_hard, no discoverable edges)', '$-$ $G_{\\text{data}}$ (hard edges only)'),
    ('- Physics prior (uniform Beta(1,1), no G_hard)', '$-$ physics prior (uniform Beta(1,1), no $G_{\\text{hard}}$)'),
    ('- Stage 2 (no structure refinement, no param re-estimation)', '$-$ Stage~2 refinement'),
    ('- Temperature Calibration', '$-$ temperature ($T=1$, evidence weight fitted)'),
    ('+ intervention channels in the deferral score', '$+$ $\\tilde{E}_d,\\tilde{E}_s$ in the deferral score'),
]


def t_ablation(T, out):
    a = pd.read_csv(T / 'ablation_matched_summary.csv').set_index('variant')
    d = pd.read_csv(T / 'ablation_matched_differences.csv')
    body = 'Variant & Acc. & ECE & AURC & $\\Delta$AURC [corrected 95\\,\\% CI] & Acc.\\ at 85\\,\\% & $\\Delta$ [corrected 95\\,\\% CI]\\\\\n\\midrule\n'
    for key, name in ABL:
        if key not in a.index:
            continue
        r = a.loc[key]
        if key == 'PICL (Full)':
            body += f"{name} & {f(r.acc_all_mean)} & {f(r.ece_mean)} & {f(r.aurc_mean)} & -- & {f(r.matched_acc_85_mean)} & --\\\\\n"
            continue
        da = d[(d.variant == key) & (d.metric == 'aurc')].iloc[0]
        dm = d[(d.variant == key) & (d.metric == 'matched_acc_85')].iloc[0]
        body += (
            f"{name} & {f(r.acc_all_mean)} & {f(r.ece_mean)} & {f(r.aurc_mean)} & {sg(da.delta)} [{sg(da.ci_corrected_lo)}, {sg(da.ci_corrected_hi)}] & "
            f"{f(r.matched_acc_85_mean)} & {sg(dm.delta)} [{sg(dm.ci_corrected_lo)}, {sg(dm.ci_corrected_hi)}]\\\\\n"
        )
    n = note(
        'Ten evaluation splits; every variant is retrained on the same splits. Differences are variant minus full model; intervals from the corrected resampled $t$-test. Accuracy at 85\\,\\% is at matched coverage.'
    )
    (out / 'tab_ablation.tex').write_text(
        wrap(body, '@{}lcccccc@{}', 'Component ablation.', 'tab:ablation', n, '\\scriptsize')
    )


def t_necessity(T, out):
    a = pd.read_csv(T / 'causal_necessity_summary.csv').set_index('model')
    ps = pd.read_csv(T / 'causal_necessity_per_seed.csv')
    ps['beta'] = ps.evidence_weight / ps.temperature
    beta = ps.groupby('model').beta.mean()
    rows = [
        ('PICL (physics-anchored hybrid graph)', 'PICL (physics-anchored hybrid graph)'),
        ('NOTEARS graph (fault->gas)', 'NOTEARS graph'),
        ('DCDI graph (fault->gas)', 'DCDI graph'),
        ('Empirical class means (no causal model)', 'Empirical class means (no graph)'),
        ('Unstructured Gaussian (fully connected, no physics)', 'Unstructured Gaussian (fully connected)'),
    ]
    body = 'Model supplying completion and evidence & ERH & SHD & Evidence alone & Acc. & AURC & Acc.\\ at 85\\,\\% & $\\beta$\\\\\n\\midrule\n'
    for key, name in rows:
        r = a.loc[key]
        body += f"{name} & {f(r.ERH_mean, 2)} & {f(r.SHD_mean, 1)} & {f(r.scm_only_acc_mean)} & {pm(r.acc_full_mean, r.acc_full_std)} & {f(r.aurc_mean)} & {f(r.matched_acc85_mean)} & {f(beta.loc[key], 2)}\\\\\n"
    n = note(
        'Ten evaluation splits; Stage~3 (completion, head inputs, fusion, deferral) is identical, only the model supplying the completion and the interventional evidence changes. NOTEARS and DCDI are fitted here on the completed training records of the second PICL pass and thresholded at $|W|>0.05$, so their graph statistics differ slightly from Table~\\ref{tab:alignment}, where they are fitted on the training records before completion. Evidence alone: accuracy of $\\arg\\max_k \\log q(k\\mid y)$. $\\beta$: mean fitted evidence weight of Eq.~\\eqref{eq:fusion}.'
    )
    (out / 'tab_necessity.tex').write_text(
        wrap(
            body,
            '@{}lccccccc@{}',
            'Replacing the causal model while keeping Stage~3 unchanged.',
            'tab:necessity',
            n,
            '\\scriptsize',
        )
    )


def t_mixed(T, out):
    d = pd.read_csv(T / 'mixed_fault_ppm_per_seed.csv')
    body = (
        'Method & \\multicolumn{2}{c}{Accepted} & \\multicolumn{2}{c}{Two-fault mixtures accepted with} & Misleading\\\\\n'
        ' & single fault & two faults & a constituent label & a third label & decisions\\\\\n\\midrule\n'
    )
    for mth in ('PICL', 'Random forest'):
        sg_ = d[(d.method == mth) & (d.kind == 'single fault')]
        tw = (
            d[(d.method == mth) & (d.kind == 'two faults')]
            .groupby('seed')[['accept', 'parent', 'third', 'misleading_rate']]
            .mean()
        )
        body += (
            f"{mth} & {f(sg_.accept.mean())} & {f(tw.accept.mean())} & {f(tw.parent.mean())} & {f(tw.third.mean())} & "
            f"{f(tw.misleading_rate.mean())}\\\\\n"
        )
    n = note(
        'Ten evaluation splits; mixtures of two complete test records of different classes, $\\lambda\\,\\mathrm{ppm}_a+(1-\\lambda)\\,\\mathrm{ppm}_b$ '
        'with $\\lambda\\in\\{0.25,0.5,0.75\\}$ (averaged), 40 record pairs per class pair and split. Thresholds for $85\\,\\%$ coverage on the calibration split of each method. '
        'Misleading decisions: mixtures accepted with a label that is neither constituent, as a fraction of all mixtures.'
    )
    (out / 'tab_mixed.tex').write_text(
        wrap(body, '@{}lccccc@{}', 'Simultaneous faults built from real records.', 'tab:mixed', n)
    )


def t_losofam(T, out):
    m = pd.read_csv(T / 'v4_loso_micro_per_seed.csv')
    rows = [
        ('PICL, unseen-source read-out', 'PICL, unseen-source read-out'),
        ('PICL in-distribution read-out', 'PICL, in-distribution read-out'),
        ('Random forest', 'Random forest'),
        ('XGBoost', 'XGBoost'),
        ('SVM', 'SVM'),
        ('ANN', 'ANN'),
    ]
    rec = (
        pd.read_csv(T / 'v6_loso_transformer_tests.csv').set_index('vs')
        if (T / 'v6_loso_transformer_tests.csv').exists()
        else None
    )
    recmap = {
        'PICL in-distribution read-out': 'PICL in-distribution',
        'Random forest': 'Random forest',
        'XGBoost': 'XGBoost',
        'SVM': 'SVM',
        'ANN': 'ANN',
    }
    body = (
        'Method & \\multicolumn{3}{c}{Sub-class} & \\multicolumn{3}{c}{Mechanism family} & \\multicolumn{2}{c}{Hierarchical rule} & \\multicolumn{2}{c}{PICL minus method (transformers)}\\\\\n'
        '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\n'
        ' & Acc. & Macro-F1 & AURC & Acc. & AURC & Acc.\\ at 85\\,\\% & Decided & Errors & $\\Delta$ family acc. & $\\Delta$ family AURC\\\\\n\\midrule\n'
    )
    for key, lab in rows:
        x = m[m.method == key]
        rc1 = rc2 = '--'
        if rec is not None and key in recmap and recmap[key] in rec.index:
            rr = rec.loc[recmap[key]]
            rc1 = f'{rr.fam_acc_diff:+.3f} [{rr.fam_acc_ci_lo:+.3f}, {rr.fam_acc_ci_hi:+.3f}]' + (
                '$^{*}$' if rr.fam_acc_ci_lo > 0 else ''
            )
            rc2 = f'{rr.fam_aurc_diff:+.3f} [{rr.fam_aurc_ci_lo:+.3f}, {rr.fam_aurc_ci_hi:+.3f}]' + (
                '$^{*}$' if rr.fam_aurc_ci_hi < 0 else ''
            )
        body += (
            f"{lab} & {f(x.acc.mean())} & {f(x.macro_f1.mean())} & {f(x.aurc.mean())} & {f(x.fam_acc.mean())} & {f(x.fam_aurc.mean())} & "
            f"{f(x.fam_acc85.mean())} & {f(x.h_cov_total.mean())} & {f(x.h_err_rate.mean())} & {rc1} & {rc2}\\\\\n"
        )
    n = note(
        'Leave-one-source-out, five seeds, micro-averaged over the five held-out sources. Family: PD $\\mid$ discharge (D1, D2) $\\mid$ thermal (T1--T3), '
        'family probability = sum of its sub-class probabilities; accuracy at 85\\,\\%: matched coverage. Hierarchical rule (Section~\\ref{sec:hier}), thresholds from the calibration split: '
        'fraction of records decided automatically and fraction decided wrongly. Last two columns: paired differences on the 638 held-out records (each a test record of one fold, predictions averaged over the five seeds) with 95\\,\\% transformer-level bootstrap intervals (361 transformers or cases resampled with all their records, 4\\,000 resamples; $^{*}$ excludes zero).'
    )
    (out / 'tab_losofam.tex').write_text(
        wrap(
            body,
            '@{}lcccccccccc@{}',
            'Transfer to an unseen source at sub-class and family level.',
            'tab:losofam',
            n,
            '\\scriptsize',
        )
    )


def t_hier(T, out):
    d = pd.read_csv(T / 'v4_hierarchical_per_seed.csv')
    rows = [
        ('PICL', False, 'flat 85 %', 'PICL, single threshold (85\\,\\%)'),
        ('PICL', False, 'hierarchical 70/95 %', 'PICL, hierarchical'),
        ('PICL', True, 'hierarchical 70/95 %', 'PICL, hierarchical + two-fault screen'),
        ('Random Forest', False, 'hierarchical 70/95 %', 'Random forest, hierarchical'),
        ('Random Forest', True, 'hierarchical 70/95 %', 'Random forest, hierarchical + screen'),
        ('XGBoost', False, 'hierarchical 70/95 %', 'XGBoost, hierarchical'),
        ('XGBoost', True, 'hierarchical 70/95 %', 'XGBoost, hierarchical + screen'),
        ('SVM', False, 'hierarchical 70/95 %', 'SVM, hierarchical'),
        ('SVM', True, 'hierarchical 70/95 %', 'SVM, hierarchical + screen'),
    ]
    body = (
        'Method and rule & \\multicolumn{5}{c}{Real test records} & \\multicolumn{3}{c}{Two-fault mixtures}\\\\\n'
        '\\cmidrule(lr){2-6}\\cmidrule(lr){7-9}\n'
        ' & Decided & Sub-class & Acc. & Errors & Wrong family & Decided & Misleading & Wrong family\\\\\n\\midrule\n'
    )
    for mth, scr, rule, lab in rows:
        x = d[(d.method == mth) & (d.screen == scr) & (d.rule == rule)]
        body += (
            f"{lab} & {f(x.cov_total.mean())} & {f(x.cov_sub.mean())} & {f(x.acc_decided.mean())} & {f(x.err_rate.mean())} & {f(x.wrong_family.mean())} & "
            f"{f(x.mix_decided.mean())} & {f(x.mix_misleading.mean())} & {f(x.mix_wrong_family.mean())}\\\\\n"
        )
        if (mth, scr, rule) == ('PICL', True, 'hierarchical 70/95 %'):
            body += '\\addlinespace[2pt]\n'
    n = note(
        'Ten evaluation splits. Decided: fraction of records given an automatic sub-class or family diagnosis (sub-class: at sub-class level); accuracy at the level decided; '
        'errors and wrong family as fractions of all records. Mixtures: $\\lambda\\,\\mathrm{ppm}_a+(1-\\lambda)\\,\\mathrm{ppm}_b$ of two complete test records of different classes, '
        '$\\lambda\\in\\{0.25,0.5,0.75\\}$, 40 record pairs per class pair and split; misleading: a sub-class that is neither constituent, or a family to which neither constituent belongs. '
        'Hierarchical rule: sub-class for 70\\,\\%, family up to 95\\,\\% of the calibration records. Screen of PICL (the same flags for every method): 10\\,\\% of the calibration records flagged and referred; every other record receives at least its family.'
    )
    (out / 'tab_hier.tex').write_text(
        wrap(body, '@{}lcccccccc@{}', 'Hierarchical decisions and simultaneous faults.', 'tab:hier', n, '\\scriptsize')
    )


def t_fewshot(T, out):
    m = pd.read_csv(T / 'v5_fewshot_micro_per_seed_rep.csv')
    pf = pd.read_csv(T / 'v5_fewshot_per_fold.csv')
    nlab = pf[pf.k > 0].groupby('k').n_labelled.mean()
    rows = [
        ('PICL, unseen-source read-out', 'PICL, unseen-source read-out'),
        ('PICL in-distribution read-out', 'PICL, in-distribution read-out'),
        ('Random forest', 'Random forest'),
        ('XGBoost', 'XGBoost'),
        ('SVM', 'SVM'),
    ]
    ks = [0, 10, 20, 40]
    hdr = ' & '.join(f'\\multicolumn{{2}}{{c}}{{{"none" if k == 0 else f"{nlab[k]:.0f}"}}}' for k in ks)
    body = (
        f'Method & {hdr} & Family acc.\\\\\n'
        + ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(4))
        + '\n'
        ' & ' + ' & '.join(['Acc. & AURC'] * 4) + ' & ($33$ records)\\\\\n\\midrule\n'
    )
    for key, lab in rows:
        cells = []
        for k in ks:
            x = m[(m.method == key) & (m.k == k)]
            cells += [f(x.acc.mean()), f(x.aurc.mean())]
        x = m[(m.method == key) & (m.k == 40)]
        body += f"{lab} & " + ' & '.join(cells) + f" & {f(x.fam_acc.mean())}\\\\\n"
    n = note(
        'Leave-one-source-out, five seeds, micro-averaged over the held-out sources. Columns: mean number of labelled records of the held-out source '
        '(one, three or six per class where available, ten draws) on which a per-class logit bias and a temperature are fitted; metrics on the remaining records of the source. '
        'Family acc.: mechanism-family accuracy with $33$ labelled records (without: Table~\\ref{tab:losofam}).'
    )
    (out / 'tab_fewshot.tex').write_text(
        wrap(
            body,
            '@{}lccccccccc@{}',
            'Local recalibration of a transferred model with a few labelled records of the new source.',
            'tab:fewshot',
            n,
            '\\scriptsize',
        )
    )


def _acc_aurc_from_scores(seeds, key):
    """Per-seed accuracy and AURC of a score-dump column prefix (e.g. 'picl_P', 'random_forest_P')."""
    out = []
    for seed in seeds:
        te = load_scores(seed, 'test')
        y = te.label.values
        P = te[[f'{key}{k}' for k in range(6)]].values
        c = P.argmax(1) == y
        out.append((c.mean(), _aurc(P.max(1), c)))
    return np.array(out)


def t_confirm(T, out):
    ev, cf = list(range(52, 62)), list(range(62, 72))
    keys = [
        ('PICL', 'picl_P', 'PICL'),
        ('Random forest', 'random_forest_P', 'Random forest'),
        ('CatBoost', None, 'CatBoost'),
        ('SVM', 'svm_P', 'SVM'),
        ('XGBoost', 'xgboost_P', 'XGBoost'),
    ]
    rec = (
        pd.read_csv(T / 'v6_indist_transformer_tests_pooled20.csv').set_index('vs')
        if (T / 'v6_indist_transformer_tests_pooled20.csv').exists()
        else None
    )
    cbe = pd.read_csv(T / 'v6_catboost_eval_per_seed.csv').set_index('seed')
    cbc = pd.read_csv(T / 'confirm' / 'v6_catboost_eval_per_seed.csv').set_index('seed')
    body = (
        'Method & \\multicolumn{2}{c}{Evaluation splits 52--61} & \\multicolumn{2}{c}{Confirmation splits 62--71} & \\multicolumn{2}{c}{PICL minus method, 20 splits pooled}\\\\\n'
        '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n & Acc. & AURC & Acc. & AURC & $\\Delta$ acc.\\ [95\\,\\% CI] & $\\Delta$ AURC [95\\,\\% CI]\\\\\n\\midrule\n'
    )
    for lab, key, rk in keys:
        if key is None:
            e = cbe.loc[ev][['accuracy', 'aurc']].values
            c = cbc.loc[cf][['accuracy', 'aurc']].values
        else:
            e, c = _acc_aurc_from_scores(ev, key), _acc_aurc_from_scores(cf, key)
        rc1 = rc2 = '--'
        if rec is not None and rk in rec.index:
            r = rec.loc[rk]
            rc1 = f'{r.acc_diff:+.3f} [{r.acc_ci_lo:+.3f}, {r.acc_ci_hi:+.3f}]' + ('$^{*}$' if r.acc_ci_lo > 0 else '')
            rc2 = f'{r.aurc_diff:+.3f} [{r.aurc_ci_lo:+.3f}, {r.aurc_ci_hi:+.3f}]' + (
                '$^{*}$' if r.aurc_ci_hi < 0 else ''
            )
        body += f"{lab} & {pm(e[:, 0].mean(), e[:, 0].std(ddof=1))} & {f(e[:, 1].mean())} & {pm(c[:, 0].mean(), c[:, 0].std(ddof=1))} & {f(c[:, 1].mean())} & {rc1} & {rc2}\\\\\n"
    n = note(
        'Full-coverage accuracy (mean $\\pm$ s.d.) and AURC on the ten evaluation splits, on which the method was scored repeatedly during its design rounds, and on ten confirmation splits scored once with the final configuration. '
        'Last two columns: transformer-level cluster bootstrap over the pooled out-of-fold predictions of all twenty splits (358 transformers or cases, 635 records, 2\\,551 predictions, 4\\,000 resamples; differences PICL minus the method; for AURC, negative favours PICL; $^{*}$ interval excludes zero).'
    )
    (out / 'tab_confirm.tex').write_text(
        wrap(body, '@{}lcccccc@{}', 'Evaluation and confirmation splits.', 'tab:confirm', n, '\\scriptsize')
    )


def t_lcurve(T, out):
    d = pd.read_csv(T / 'learning_curve_per_seed.csv')
    meths = [
        ('PICL', 'PICL'),
        ('PICL, uniform prior', 'uniform prior'),
        ('PICL without SCM evidence', 'no evidence'),
        ('XGBoost', 'XGBoost'),
    ]
    body = (
        'Training & $n$ & \\multicolumn{4}{c}{Accuracy} & \\multicolumn{4}{c}{AURC}\\\\\n'
        'fraction & & '
        + ' & '.join(m[1] for m in meths)
        + ' & '
        + ' & '.join(m[1] for m in meths)
        + '\\\\\n\\midrule\n'
    )
    for fr in sorted(d.fraction.unique()):
        x = d[d.fraction == fr]
        acc = [x[x.method == m].acc.mean() for m, _ in meths]
        au = [x[x.method == m].aurc.mean() for m, _ in meths]
        body += (
            f"{int(round(100 * fr))}\\,\\% & {x.n_train.mean():.0f} & "
            + ' & '.join(f(a) for a in acc)
            + ' & '
            + ' & '.join(f(a) for a in au)
            + '\\\\\n'
        )
    n = note(
        'Ten evaluation splits; the training partition is subsampled (stratified by class), calibration and test partitions are unchanged, and every method is retrained. '
        'Uniform prior: no hard edges, every discoverable edge $\\mathrm{Beta}(1,1)$. No evidence: the same read-out without the SCM evidence (random forest on gases and ratios). XGBoost on gases and ratios, class-balanced, temperature-scaled.'
    )
    (out / 'tab_lcurve.tex').write_text(
        wrap(
            body,
            '@{}lrcccccccc@{}',
            'Sample efficiency: training-set size and the contribution of the causal model.',
            'tab:lcurve',
            n,
            '\\scriptsize',
        )
    )


def main(T, out):
    out.mkdir(parents=True, exist_ok=True)
    for fn in (
        t_data,
        t_klsens,
        t_alignment,
        t_cf,
        t_design,
        t_main,
        t_perclass,
        t_paired,
        t_selective,
        t_gate,
        t_residuals,
        t_scmvariants,
        t_loso,
        t_weak,
        t_missing,
        t_nonfault,
        t_cost,
        t_ablation,
        t_necessity,
        t_mixed,
        t_lcurve,
        t_losofam,
        t_hier,
        t_fewshot,
        t_confirm,
    ):
        try:
            fn(T, out)
            print('ok  ', fn.__name__)
        except Exception as e:
            print('FAIL', fn.__name__, type(e).__name__, e)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tables', default='tables')
    ap.add_argument('--out', default='tex_tables')
    a = ap.parse_args()
    main(Path(a.tables), Path(a.out))
