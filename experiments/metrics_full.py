"""Classification metrics, per-class metrics, confusion matrices, reliability bins and paired tests.

python experiments/metrics_full.py --seeds 52 ... 61
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import EVAL_SEEDS, cols, load_meta, load_scores

FAULTS = ['PD', 'D1', 'D2', 'T1', 'T2', 'T3']
LABELS = list(range(6))
METHODS = [
    ('picl', 'PICL (all samples)'),
    ('random_forest', 'Random forest'),
    ('xgboost', 'XGBoost'),
    ('svm', 'SVM'),
    ('ann', 'ANN'),
    ('random_forest_gases_only', 'Random forest (gases only)'),
]
N_TEST_OVER_N_TRAIN = 0.2 / 0.6


def all_metrics(y, pred, P=None):
    m = dict(
        accuracy=float((pred == y).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average='macro', labels=LABELS, zero_division=0)),
        weighted_f1=float(f1_score(y, pred, average='weighted', labels=LABELS, zero_division=0)),
        mcc=float(matthews_corrcoef(y, pred)),
        cohen_kappa=float(cohen_kappa_score(y, pred)),
    )
    if P is not None:
        P = np.clip(P, 1e-12, 1.0)
        P = P / P.sum(1, keepdims=True)
        m['log_loss'] = float(log_loss(y, P, labels=LABELS))
        m['brier_multiclass'] = float(((P - np.eye(6)[y]) ** 2).sum(1).mean())
    return m


def nb_corrected_t(diff, ratio=N_TEST_OVER_N_TRAIN):
    """Nadeau & Bengio (2003) corrected resampled t-test on split-level differences."""
    d = np.asarray(diff, float)
    J = len(d)
    sd = d.std(ddof=1)
    if sd == 0:
        return float('nan'), float('nan'), (d.mean(), d.mean())
    se = sd * np.sqrt(1.0 / J + ratio)
    t = d.mean() / se
    p = 2 * stats.t.sf(abs(t), J - 1)
    half = stats.t.ppf(0.975, J - 1) * se
    return float(t), float(p), (float(d.mean() - half), float(d.mean() + half))


def ece_bins(conf, correct, n_bins=15):
    rows, ece = [], 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        sel = (conf > lo) & (conf <= hi)
        if sel.sum() == 0:
            rows.append(dict(bin_lo=lo, bin_hi=hi, n=0, mean_conf=np.nan, accuracy=np.nan))
            continue
        a, c = float(correct[sel].mean()), float(conf[sel].mean())
        ece += sel.sum() / len(conf) * abs(a - c)
        rows.append(dict(bin_lo=lo, bin_hi=hi, n=int(sel.sum()), mean_conf=c, accuracy=a))
    return rows, float(ece)


def main(seeds, out_dir, score_dir):
    per_seed, per_class_rows, rel_rows, ece_rows = [], [], [], []
    cm_total = {}
    for seed in seeds:
        te = load_scores(seed, 'test', score_dir)
        meta = load_meta(seed, score_dir)
        y = te['label'].to_numpy()
        for key, name in METHODS:
            P = cols(te, f'{key}_P')
            pred = P.argmax(1)
            per_seed.append(dict(seed=seed, method=name, coverage=1.0, **all_metrics(y, pred, P)))
            if key in ('picl', 'random_forest', 'xgboost'):
                cm_total.setdefault(name, np.zeros((6, 6), int))
                cm_total[name] += confusion_matrix(y, pred, labels=LABELS)
                Pr, R, F, S = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
                for k in range(6):
                    per_class_rows.append(
                        dict(
                            seed=seed,
                            method=name,
                            fault=FAULTS[k],
                            n=int(S[k]),
                            precision=float(Pr[k]),
                            recall=float(R[k]),
                            f1=float(F[k]),
                        )
                    )
            if key == 'picl':
                acc_mask = P.max(1) >= float(meta['threshold'])
                per_seed.append(
                    dict(
                        seed=seed,
                        method='PICL (accepted only)',
                        coverage=float(acc_mask.mean()),
                        accuracy=float((pred[acc_mask] == y[acc_mask]).mean()),
                    )
                )
                bins, ece = ece_bins(P.max(1), pred == y)
                rel_rows += [dict(seed=seed, **b) for b in bins]
                Praw = cols(te, 'picl_Praw')
                _, ece_raw = ece_bins(Praw.max(1), Praw.argmax(1) == y)
                ece_rows.append(
                    dict(
                        seed=seed,
                        ece=ece,
                        ece_uncalibrated=ece_raw,
                        temperature=meta['temperature'],
                        evidence_weight=meta.get('evidence_weight', 0.0),
                    )
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    ps = pd.DataFrame(per_seed)
    ps.to_csv(out_dir / 'metrics_per_seed.csv', index=False)
    pd.DataFrame(per_class_rows).to_csv(out_dir / 'per_class_metrics.csv', index=False)
    pd.DataFrame(rel_rows).to_csv(out_dir / 'reliability_bins.csv', index=False)
    pd.DataFrame(ece_rows).to_csv(out_dir / 'calibration_per_seed.csv', index=False)
    for name, cm in cm_total.items():
        tag = {'PICL (all samples)': 'PICL', 'Random forest': 'RandomForest', 'XGBoost': 'XGBoost'}[name]
        pd.DataFrame(cm, index=FAULTS, columns=FAULTS).to_csv(out_dir / f'confusion_matrix_{tag}.csv')

    sig = []
    ref = ps[ps.method == 'PICL (all samples)'].set_index('seed')
    for _, name in METHODS[1:]:
        oth = ps[ps.method == name].set_index('seed')
        for metric in ('accuracy', 'macro_f1', 'mcc'):
            a, b = ref[metric].to_numpy(), oth.loc[ref.index, metric].to_numpy()
            d = a - b
            t, pt_ = stats.ttest_rel(a, b)
            try:
                _, pw = stats.wilcoxon(a, b)
            except ValueError:
                pw = np.nan
            tn, pn, ci = nb_corrected_t(d)
            sig.append(
                dict(
                    comparison=f'PICL vs {name}',
                    metric=metric,
                    mean_diff=float(d.mean()),
                    wins=int((d > 0).sum()),
                    n_splits=len(d),
                    p_ttest=float(pt_),
                    p_wilcoxon=float(pw),
                    p_corrected=pn,
                    ci95_corrected_lo=ci[0],
                    ci95_corrected_hi=ci[1],
                )
            )
    sigdf = pd.DataFrame(sig)
    sigdf.to_csv(out_dir / 'significance_tests.csv', index=False)
    pd.set_option('display.width', 250)
    cols_ = ['accuracy', 'balanced_accuracy', 'macro_f1', 'mcc', 'log_loss', 'brier_multiclass', 'coverage']
    print(ps.groupby('method')[cols_].agg(['mean', 'std']).round(4).to_string())
    print(sigdf.round(4).to_string(index=False))
    print(pd.DataFrame(ece_rows).mean().round(4).to_string())
    print(f'Written to {out_dir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=EVAL_SEEDS)
    ap.add_argument('--out', default='tables')
    ap.add_argument('--score-dir', default='results/scores')
    a = ap.parse_args()
    main(a.seeds, Path(a.out), a.score_dir)
