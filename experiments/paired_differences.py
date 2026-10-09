"""Paired differences between PICL and every other method with the corrected resampled t-test.

python experiments/paired_differences.py --seeds 52 ... 61
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import EVAL_SEEDS, TARGET_COVERAGE, accepted_acc_at_coverage, aurc, load_scores, method_scores
from metrics_full import nb_corrected_t

PAIRS = [
    ('picl', 'random_forest', 'PICL - random forest'),
    ('picl', 'xgboost', 'PICL - XGBoost'),
    ('picl', 'svm', 'PICL - SVM'),
    ('picl', 'ann', 'PICL - ANN'),
    ('picl', 'random_forest_gases_only', 'PICL - random forest (gases only)'),
    ('picl', 'picl_head', 'PICL - PICL head without SCM evidence'),
    ('picl', 'picl_composite', 'PICL - PICL with intervention channels in the gate'),
]
LOWER_IS_BETTER = {'aurc'}


def stats_of(conf, pred, y):
    corr = pred == y
    return dict(
        acc=float(corr.mean()),
        macro_f1=float(f1_score(y, pred, average='macro', labels=list(range(6)), zero_division=0)),
        aurc=aurc(conf, corr),
        acc85=accepted_acc_at_coverage(conf, corr, TARGET_COVERAGE)['accepted_acc'],
    )


def main(seeds, out_dir, score_dir):
    rows = []
    for a, b_, label in PAIRS:
        d = {k: [] for k in ('acc', 'macro_f1', 'aurc', 'acc85')}
        for seed in seeds:
            te = load_scores(seed, 'test', score_dir)
            y = te['label'].to_numpy()
            sa = stats_of(*method_scores(te, a), y)
            sb = stats_of(*method_scores(te, b_), y)
            for k in d:
                d[k].append(sa[k] - sb[k])
        for k, v in d.items():
            v = np.asarray(v)
            _, p_c, ci = nb_corrected_t(v)
            t_half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            wins = int((v < 0).sum()) if k in LOWER_IS_BETTER else int((v > 0).sum())
            rows.append(
                dict(
                    comparison=label,
                    metric=k,
                    mean_diff=float(v.mean()),
                    ci_corrected_lo=ci[0],
                    ci_corrected_hi=ci[1],
                    p_corrected=p_c,
                    t_ci_lo=float(v.mean() - t_half),
                    t_ci_hi=float(v.mean() + t_half),
                    p_ttest=float(stats.ttest_1samp(v, 0.0).pvalue),
                    wins=wins,
                    n_splits=len(v),
                    range_lo=float(v.min()),
                    range_hi=float(v.max()),
                )
            )
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'paired_differences.csv', index=False)
    pd.set_option('display.width', 250)
    print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=EVAL_SEEDS)
    ap.add_argument('--out', default='tables')
    ap.add_argument('--score-dir', default='results/scores')
    a = ap.parse_args()
    main(a.seeds, Path(a.out), a.score_dir)
