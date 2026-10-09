"""Transformer-level paired bootstrap (4000 resamples) for the in-distribution results.

PYTHONPATH=.:experiments/v4 python experiments/v6/transformer_tests.py [--tag _pooled20]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v4'))
from core import aurc

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='')
ap.add_argument('--n-boot', type=int, default=4000)
ap.add_argument('--seed', type=int, default=17624)
a = ap.parse_args()

prov = pd.read_csv('data/dga_provenance.csv')[['sample_id', 'case_id']]
R = pd.read_csv(f'tables/v6_indist_pooled_predictions{a.tag}.csv').merge(
    prov, on='sample_id', how='left', validate='many_to_one'
)
assert R.case_id.notna().all()
methods = ['Random forest', 'CatBoost', 'SVM', 'XGBoost']
cases = R.case_id.unique()
groups = [np.flatnonzero(R.case_id.values == c) for c in cases]
y = R.y.values
corr = {m: (R[f'{m}|pred'].values == y) for m in ['PICL'] + methods}
conf = {m: R[f'{m}|conf'].values for m in ['PICL'] + methods}
rng = np.random.default_rng(a.seed)
idx_sets = [np.concatenate([groups[i] for i in rng.integers(0, len(cases), len(cases))]) for _ in range(a.n_boot)]
rows = []
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
            n_cases=len(cases),
            n_records=R.sample_id.nunique(),
            n_predictions=len(R),
        )
    )
o = pd.DataFrame(rows)
pd.set_option('display.width', 220)
print(o.round(4).to_string())
o.to_csv(f'tables/v6_indist_transformer_tests{a.tag}.csv', index=False)
