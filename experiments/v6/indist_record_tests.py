"""Record-level bootstrap over the pooled out-of-fold predictions of several splits.

python experiments/v6/indist_record_tests.py --seeds 52 ... 61
"""

import sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v4'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v5'))
from core import load, aurc, base_X, proba6, temp_fit, temp_apply
from records import split_frames
from catboost import CatBoostClassifier

import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', type=int, nargs='+', default=list(range(52, 62)))
ap.add_argument('--tag', default='')
a = ap.parse_args()
SEEDS = a.seeds
recs = []
for seed in SEEDS:
    root = 'results/v4/indist' if seed < 62 else 'results/v4_confirm/indist'
    d = load(f'{root}/seed_{seed}.npz')
    fr = split_frames(seed)
    ids = fr['te']['sample_id'].to_numpy()
    y = d['te_y']
    df = pd.read_csv('results/scores/scores_test.csv')
    df = df[df.seed == seed].drop(columns='seed').reset_index(drop=True)
    assert (df.label.values == y).all()
    P = {'PICL': d['te_P']}
    for key, name in (('random_forest_P', 'Random forest'), ('svm_P', 'SVM'), ('xgboost_P', 'XGBoost')):
        P[name] = df[[f'{key}{k}' for k in range(6)]].values
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
    P['CatBoost'] = temp_apply(proba6(cb, Xb['te']), T)
    for i in range(len(y)):
        row = dict(seed=seed, sample_id=ids[i], y=y[i])
        for m, Pm in P.items():
            row[f'{m}|pred'] = int(Pm[i].argmax())
            row[f'{m}|conf'] = float(Pm[i].max())
        recs.append(row)
    print(seed, flush=True)
R = pd.DataFrame(recs)
R.to_csv(f'tables/v6_indist_pooled_predictions{a.tag}.csv', index=False)
methods = ['Random forest', 'CatBoost', 'SVM', 'XGBoost']
ids = R.sample_id.unique()
groups = {sid: np.where(R.sample_id.values == sid)[0] for sid in ids}
rng = np.random.default_rng(0)
y = R.y.values
corr = {m: (R[f'{m}|pred'].values == y) for m in ['PICL'] + methods}
conf = {m: R[f'{m}|conf'].values for m in ['PICL'] + methods}
rows = []
idx_sets = [np.concatenate([groups[s] for s in rng.choice(ids, len(ids))]) for _ in range(4000)]
for m in methods:
    da = [corr['PICL'][ix].mean() - corr[m][ix].mean() for ix in idx_sets]
    du = [aurc(conf['PICL'][ix], corr['PICL'][ix]) - aurc(conf[m][ix], corr[m][ix]) for ix in idx_sets]
    rows.append(
        dict(
            vs=m,
            acc_diff=corr['PICL'].mean() - corr[m].mean(),
            acc_ci_lo=np.quantile(da, 0.025),
            acc_ci_hi=np.quantile(da, 0.975),
            aurc_diff=aurc(conf['PICL'], corr['PICL']) - aurc(conf[m], corr[m]),
            aurc_ci_lo=np.quantile(du, 0.025),
            aurc_ci_hi=np.quantile(du, 0.975),
            n_records=len(ids),
            n_predictions=len(R),
        )
    )
o = pd.DataFrame(rows)
pd.set_option('display.width', 220)
print(o.round(4).to_string())
o.to_csv(f'tables/v6_indist_record_tests{a.tag}.csv', index=False)
print(
    'pooled accuracy:',
    {m: round(float(corr[m].mean()), 4) for m in corr},
    'pooled AURC:',
    {m: round(aurc(conf[m], corr[m]), 4) for m in corr},
)
print(R.groupby('sample_id').size().value_counts().to_dict())
