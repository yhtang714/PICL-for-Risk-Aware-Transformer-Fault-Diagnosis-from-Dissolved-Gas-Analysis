"""AURC of the composite deferral score over the simplex of channel weights.

python experiments/weight_sensitivity.py --seeds 52 ... 61
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    EVAL_SEEDS,
    TARGET_COVERAGE,
    accept_stats,
    accepted_acc_at_coverage,
    aurc,
    cols,
    composite_from,
    load_scores,
    threshold_for_coverage,
)


def simplex_grid(step=0.1):
    n = int(round(1 / step))
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            pts.append((round(i * step, 2), round(j * step, 2), round(k * step, 2)))
    return pts


def learned_combiner(cal, test):
    """Logistic-regression stacker on the three channel values of the argmax class
    plus the full channel vectors, fitted on the calibration split to predict
    correctness of the classifier's top class.  Returns a confidence for test."""
    from sklearn.linear_model import LogisticRegression

    def feats(df):
        P = cols(df, 'picl_P')
        Es = cols(df, 'picl_Esu')
        Ed = df['picl_Edu'].to_numpy()
        top = P.argmax(1)
        idx = np.arange(len(df))
        f = np.column_stack([P[idx, top], Es[idx, top], Ed, np.sort(P, axis=1)[:, -2], np.sort(Es, axis=1)[:, -2]])
        return f, top

    fc, top_c = feats(cal)
    ft, top_t = feats(test)
    y = (top_c == cal['label'].to_numpy()).astype(int)
    lr = LogisticRegression(max_iter=5000, C=1.0).fit(fc, y)
    return lr.predict_proba(ft)[:, 1], top_t, lr.predict_proba(fc)[:, 1], top_c


def main(seeds, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        cal, test = load_scores(seed, 'cal'), load_scores(seed, 'test')
        y_c, y_t = cal['label'].to_numpy(), test['label'].to_numpy()
        Pc, Pt = cols(cal, 'picl_P'), cols(test, 'picl_P')
        Edc, Edt = cal['picl_Edu'].to_numpy()[:, None], test['picl_Edu'].to_numpy()[:, None]
        Esc, Est = cols(cal, 'picl_Esu'), cols(test, 'picl_Esu')
        for w in simplex_grid(0.1) + [(1 / 3, 1 / 3, 1 / 3)]:
            Sc, St = composite_from(Pc, Edc, Esc, w), composite_from(Pt, Edt, Est, w)
            cc, pc = Sc.max(1), Sc.argmax(1)
            ct, pt = St.max(1), St.argmax(1)
            corr = pt == y_t
            thr = threshold_for_coverage(cc, TARGET_COVERAGE)
            tr = accept_stats(ct, corr, thr)
            ma = accepted_acc_at_coverage(ct, corr, TARGET_COVERAGE)
            rows.append(
                dict(
                    seed=seed,
                    w_p=w[0],
                    w_d=w[1],
                    w_s=w[2],
                    kind='grid' if w != (1 / 3, 1 / 3, 1 / 3) else 'equal',
                    acc_full=float(corr.mean()),
                    aurc=aurc(ct, corr),
                    transferred_coverage=tr['coverage'],
                    transferred_acc=tr['accepted_acc'],
                    matched_acc85=ma['accepted_acc'],
                )
            )
        conf_t, pred_t, conf_c, pred_c = learned_combiner(cal, test)
        corr = pred_t == y_t
        thr = threshold_for_coverage(conf_c, TARGET_COVERAGE)
        tr = accept_stats(conf_t, corr, thr)
        ma = accepted_acc_at_coverage(conf_t, corr, TARGET_COVERAGE)
        rows.append(
            dict(
                seed=seed,
                w_p=np.nan,
                w_d=np.nan,
                w_s=np.nan,
                kind='learned (logistic stacker)',
                acc_full=float(corr.mean()),
                aurc=aurc(conf_t, corr),
                transferred_coverage=tr['coverage'],
                transferred_acc=tr['accepted_acc'],
                matched_acc85=ma['accepted_acc'],
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'weight_sensitivity_per_seed.csv', index=False)
    agg = (
        df.groupby(['kind', 'w_p', 'w_d', 'w_s'], dropna=False)
        .agg(
            acc_full=('acc_full', 'mean'),
            aurc=('aurc', 'mean'),
            aurc_sd=('aurc', 'std'),
            matched_acc85=('matched_acc85', 'mean'),
            matched_acc85_sd=('matched_acc85', 'std'),
            transferred_coverage=('transferred_coverage', 'mean'),
            transferred_acc=('transferred_acc', 'mean'),
            n_seeds=('seed', 'nunique'),
        )
        .reset_index()
    )
    agg.to_csv(out_dir / 'weight_sensitivity_summary.csv', index=False)

    g = agg[agg.kind == 'grid'].copy()
    ref = g[(g.w_p == 0.4) & (g.w_d == 0.3) & (g.w_s == 0.3)].iloc[0]
    conf_only = g[(g.w_p == 1.0)].iloc[0]
    best = g.sort_values('aurc').iloc[0]
    print('\n=== Weight sensitivity (mean over seeds) ===')
    print(f'reference (0.4,0.3,0.3): AURC {ref.aurc:.5f}  acc@85 {ref.matched_acc85:.4f}  acc_full {ref.acc_full:.4f}')
    print(f'confidence only (1,0,0): AURC {conf_only.aurc:.5f}  acc@85 {conf_only.matched_acc85:.4f}')
    print(
        f'best grid point       : ({best.w_p},{best.w_d},{best.w_s}) AURC {best.aurc:.5f}  acc@85 {best.matched_acc85:.4f}'
    )
    within = g[g.aurc <= ref.aurc + ref.aurc_sd]
    print(f'grid points with AURC within one seed-sd of the reference: {len(within)}/{len(g)}')
    better = g[g.aurc < conf_only.aurc]
    print(f'grid points beating confidence-only AURC: {len(better)}/{len(g)}')
    print(f'w_p range of those points: {better.w_p.min()} .. {better.w_p.max()}')
    lr = agg[agg.kind.str.startswith('learned')].iloc[0]
    print(f'learned combiner      : AURC {lr.aurc:.5f}  acc@85 {lr.matched_acc85:.4f}')
    print('\nfull grid (AURC):')
    print(g.pivot_table(index='w_p', columns='w_d', values='aurc').round(5).to_string())
    print(f'\nWritten to {out_dir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=EVAL_SEEDS)
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    main(a.seeds, Path(a.out))
