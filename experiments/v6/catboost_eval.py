"""CatBoost baseline (gases and ratios, class-balanced, temperature-scaled) on the exported splits.

python experiments/v6/catboost_eval.py --seeds 52 ... 61 --root results/v4/indist --out tables
"""

import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v4'))
from core import load, base_X, proba6, temp_fit, temp_apply, aurc, acc_at, thr_for, K
from catboost import CatBoostClassifier
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', type=int, nargs='+', default=list(range(52, 62)))
ap.add_argument('--root', default='results/v4/indist')
ap.add_argument('--out', default='tables')
a = ap.parse_args()
rows = []
for seed in a.seeds:
    d = load(f'{a.root}/seed_{seed}.npz')
    Xb = {s: base_X(d[f'{s}_gas_c'], d[f'{s}_lp_c']) for s in ('tr', 'ca', 'te')}
    cb = CatBoostClassifier(
        iterations=500,
        depth=4,
        learning_rate=0.05,
        random_seed=seed,
        verbose=0,
        auto_class_weights='Balanced',
        thread_count=2,
    ).fit(Xb['tr'], d['tr_y'])
    Pc = proba6(cb, Xb['ca'])
    T = temp_fit(Pc, d['ca_y'])
    Pc = temp_apply(Pc, T)
    Pt = temp_apply(proba6(cb, Xb['te']), T)
    y = d['te_y']
    pred = Pt.argmax(1)
    corr = pred == y
    oh = np.eye(K)[y]
    t = thr_for(Pc.max(1))
    m = Pt.max(1) >= t
    rows.append(
        dict(
            seed=seed,
            method='CatBoost',
            accuracy=corr.mean(),
            balanced_accuracy=balanced_accuracy_score(y, pred),
            macro_f1=f1_score(y, pred, average='macro', labels=list(range(K)), zero_division=0),
            mcc=matthews_corrcoef(y, pred),
            brier_multiclass=((Pt - oh) ** 2).sum(1).mean(),
            aurc=aurc(Pt.max(1), corr),
            acc85=acc_at(Pt.max(1), corr),
            cov_transferred=m.mean(),
            acc_transferred=corr[m].mean(),
            T=T,
        )
    )
    print(seed, flush=True)
pd.DataFrame(rows).to_csv(Path(a.out) / 'v6_catboost_eval_per_seed.csv', index=False)
print(pd.DataFrame(rows).mean(numeric_only=True).round(4))
