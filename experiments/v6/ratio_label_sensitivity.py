"""Accuracy on records whose sub-class was assigned by a gas-ratio rule versus the rest.

python experiments/v6/ratio_label_sensitivity.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--pred', default='tables/v6_indist_pooled_predictions.csv')
ap.add_argument('--prov', default='data/dga_provenance.csv')
ap.add_argument('--out', default='tables')
a = ap.parse_args()

pred = pd.read_csv(a.pred)
prov = pd.read_csv(a.prov).set_index('sample_id')
pred['ratio_label'] = pred['sample_id'].map(prov['label_basis'].str.contains('ratio', case=False)).astype(bool)
methods = sorted({c.split('|')[0] for c in pred.columns if c.endswith('|pred')})
rows = []
for subset, sel in (
    ('all', pred.index == pred.index),
    ('inspection-confirmed sub-class', ~pred['ratio_label']),
    ('ratio-derived sub-class', pred['ratio_label']),
):
    d = pred[sel]
    for m in methods:
        rows.append(
            dict(
                subset=subset,
                method=m,
                n_predictions=len(d),
                n_records=d['sample_id'].nunique(),
                share_of_predictions=len(d) / len(pred),
                accuracy=(d[f'{m}|pred'] == d['y']).mean(),
            )
        )
res = pd.DataFrame(rows)
Path(a.out).mkdir(parents=True, exist_ok=True)
res.to_csv(Path(a.out) / 'v6_ratio_label_sensitivity.csv', index=False)
pd.set_option('display.width', 200)
print(res.pivot(index='method', columns='subset', values='accuracy').round(4).to_string())
print(res.groupby('subset')[['n_predictions', 'n_records', 'share_of_predictions']].first().to_string())
