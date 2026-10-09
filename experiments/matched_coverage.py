"""Risk-coverage curves, AURC, accepted accuracy at matched coverage and cost under the cost rule, for PICL and the calibrated baselines.

python experiments/matched_coverage.py --seeds 52 ... 61
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    COST_RATIOS,
    COVERAGE_GRID,
    EVAL_SEEDS,
    accept_stats,
    accepted_acc_at_coverage,
    load_scores,
    method_scores,
    min_cost_threshold,
    risk_coverage,
    threshold_for_coverage,
)

METHODS = [
    ('picl', 'PICL'),
    ('picl_head', 'PICL head without SCM evidence'),
    ('picl_composite', 'PICL + intervention channels'),
    ('random_forest', 'Random forest'),
    ('xgboost', 'XGBoost'),
    ('svm', 'SVM'),
    ('ann', 'ANN'),
    ('random_forest_gases_only', 'Random forest (gases only)'),
    ('random_forest+causal', 'Random forest + intervention channels'),
    ('xgboost+causal', 'XGBoost + intervention channels'),
    ('svm+causal', 'SVM + intervention channels'),
]


def main(seeds, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, cost_rows, aurc_rows, curves = [], [], [], []
    grid = np.linspace(0.02, 1.0, 50)
    for seed in seeds:
        cal = load_scores(seed, 'cal')
        test = load_scores(seed, 'test')
        y_cal, y_te = cal['label'].to_numpy(), test['label'].to_numpy()
        for key, name in METHODS:
            cc, pc = method_scores(cal, key)
            ct, pt = method_scores(test, key)
            corr_cal, corr_te = pc == y_cal, pt == y_te
            cov_curve, risk_curve, a = risk_coverage(ct, corr_te)
            aurc_rows.append(dict(seed=seed, method=name, key=key, aurc=a, acc_full=float(corr_te.mean())))
            curves.append(
                pd.DataFrame(
                    dict(seed=seed, method=name, key=key, coverage=grid, risk=np.interp(grid, cov_curve, risk_curve))
                )
            )
            for c in COVERAGE_GRID:
                thr = threshold_for_coverage(cc, c)
                st_tr = accept_stats(ct, corr_te, thr)
                st_ma = accepted_acc_at_coverage(ct, corr_te, c)
                rows.append(
                    dict(
                        seed=seed,
                        method=name,
                        key=key,
                        target_coverage=c,
                        transferred_coverage=st_tr['coverage'],
                        transferred_accepted_acc=st_tr['accepted_acc'],
                        transferred_n_errors=st_tr['n_errors'],
                        transferred_n_deferred=st_tr['n_deferred'],
                        matched_accepted_acc=st_ma['accepted_acc'],
                        matched_n_errors=st_ma['n_errors'],
                        matched_n_deferred=st_ma['n_deferred'],
                    )
                )
            n_err_full = int((~corr_te).sum())
            for r in COST_RATIOS:
                thr = min_cost_threshold(cc, corr_cal, r)
                st = accept_stats(ct, corr_te, thr)
                cost = r * st['n_errors'] + st['n_deferred']
                base_cost = r * n_err_full
                cost_rows.append(
                    dict(
                        seed=seed,
                        method=name,
                        key=key,
                        r=r,
                        threshold=thr,
                        coverage=st['coverage'],
                        accepted_acc=st['accepted_acc'],
                        n_errors=st['n_errors'],
                        n_deferred=st['n_deferred'],
                        expected_cost=cost,
                        cost_no_gate=base_cost,
                        cost_reduction=(base_cost - cost) / base_cost if base_cost else 0.0,
                    )
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'matched_coverage_per_seed.csv', index=False)
    agg = (
        df.groupby(['method', 'key', 'target_coverage'])
        .agg(
            transferred_coverage=('transferred_coverage', 'mean'),
            transferred_acc=('transferred_accepted_acc', 'mean'),
            transferred_acc_sd=('transferred_accepted_acc', 'std'),
            transferred_errors=('transferred_n_errors', 'mean'),
            transferred_deferred=('transferred_n_deferred', 'mean'),
            matched_acc=('matched_accepted_acc', 'mean'),
            matched_acc_sd=('matched_accepted_acc', 'std'),
            matched_errors=('matched_n_errors', 'mean'),
            matched_deferred=('matched_n_deferred', 'mean'),
            n_seeds=('seed', 'nunique'),
        )
        .reset_index()
    )
    agg.to_csv(out_dir / 'matched_coverage_summary.csv', index=False)

    dc = pd.DataFrame(cost_rows)
    dc.to_csv(out_dir / 'matched_cost_per_seed.csv', index=False)
    cagg = (
        dc.groupby(['method', 'key', 'r'])
        .agg(
            threshold=('threshold', 'mean'),
            coverage=('coverage', 'mean'),
            accepted_acc=('accepted_acc', 'mean'),
            n_errors=('n_errors', 'mean'),
            n_deferred=('n_deferred', 'mean'),
            expected_cost=('expected_cost', 'mean'),
            cost_no_gate=('cost_no_gate', 'mean'),
            cost_reduction=('cost_reduction', 'mean'),
            cost_reduction_sd=('cost_reduction', 'std'),
        )
        .reset_index()
    )
    cagg.to_csv(out_dir / 'matched_cost_summary.csv', index=False)

    da = pd.DataFrame(aurc_rows)
    da.to_csv(out_dir / 'aurc_all_methods_per_seed.csv', index=False)
    aagg = (
        da.groupby(['method', 'key'])
        .agg(
            aurc=('aurc', 'mean'),
            aurc_sd=('aurc', 'std'),
            acc_full=('acc_full', 'mean'),
            acc_full_sd=('acc_full', 'std'),
            n_seeds=('seed', 'nunique'),
        )
        .reset_index()
        .sort_values('aurc')
    )
    aagg.to_csv(out_dir / 'aurc_all_methods.csv', index=False)
    pd.concat(curves).to_csv(out_dir / 'risk_coverage_curves_all.csv', index=False)

    from metrics_full import nb_corrected_t

    def paired(k1, k2, col, sub, label):
        a = sub[sub.key == k1].set_index('seed')[col]
        b = sub[sub.key == k2].set_index('seed')[col]
        b = b.loc[a.index]
        t, p = stats.ttest_rel(a, b)
        try:
            w = stats.wilcoxon(a, b).pvalue
        except ValueError:
            w = float('nan')
        _, pc, ci = nb_corrected_t((a - b).to_numpy())
        better = int(((a - b) < 0).sum()) if col == 'aurc' else int(((a - b) > 0).sum())
        return dict(
            comparison=label,
            metric=col,
            mean_a=a.mean(),
            mean_b=b.mean(),
            diff=(a - b).mean(),
            diff_sd=(a - b).std(),
            a_better=better,
            t=t,
            p_ttest=p,
            p_wilcoxon=w,
            p_corrected=pc,
            ci_corrected_lo=ci[0],
            ci_corrected_hi=ci[1],
            n=len(a),
        )

    tests = []
    sub85 = df[df.target_coverage == 0.85]
    for k2, lab in (
        ('picl_head', 'head without SCM evidence'),
        ('picl_composite', '+ intervention channels'),
        ('random_forest', 'RF'),
        ('xgboost', 'XGBoost'),
        ('svm', 'SVM'),
        ('ann', 'ANN'),
        ('random_forest_gases_only', 'RF (gases only)'),
    ):
        tests.append(paired('picl', k2, 'aurc', da, f'PICL vs {lab}'))
        tests.append(paired('picl', k2, 'matched_accepted_acc', sub85, f'PICL vs {lab} @85% matched'))
        tests.append(paired('picl', k2, 'transferred_accepted_acc', sub85, f'PICL vs {lab} @85% transferred'))
    for b_ in ('xgboost', 'random_forest', 'svm'):
        tests.append(paired(f'{b_}+causal', b_, 'aurc', da, f'{b_}+intervention channels vs {b_}'))
    pd.DataFrame(tests).to_csv(out_dir / 'matched_coverage_tests.csv', index=False)

    pd.set_option('display.width', 200)
    print('\n=== AURC and full-coverage accuracy (all methods, same inputs) ===')
    print(aagg.round(5).to_string(index=False))
    print('\n=== Accepted accuracy at matched coverage (mean over seeds) ===')
    piv = agg.pivot(index='method', columns='target_coverage', values='matched_acc')
    print(piv.round(4).to_string())
    print('\n=== Errors among accepted at matched coverage (mean over seeds, of n_test) ===')
    print(agg.pivot(index='method', columns='target_coverage', values='matched_errors').round(1).to_string())
    print('\n=== Cost reduction vs no gate (threshold chosen on calibration split) ===')
    print(cagg.pivot(index='method', columns='r', values='cost_reduction').round(3).to_string())
    print('\n=== Paired tests ===')
    print(pd.DataFrame(tests).round(5).to_string(index=False))
    print(f'\nWritten to {out_dir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=EVAL_SEEDS)
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    main(a.seeds, Path(a.out))
