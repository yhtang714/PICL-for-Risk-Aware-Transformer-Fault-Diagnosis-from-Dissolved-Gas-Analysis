"""Record-level McNemar and paired-bootstrap tests for the leave-one-source-out results.

PYTHONPATH=.:experiments/v4 python experiments/v6/loso_record_tests.py
"""

import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v4'))
from core import load, list_arrays, aurc
from eval_loso import unseen_source_readout, baselines
from picl.risk import AGG, FAMILY

SEEDS = [52, 53, 54, 55, 56]
acc_P = {}
ys = {}
for seed in SEEDS:
    for fn in list_arrays('results/v4/loso', f'seed{seed}_*'):
        d = load(fn)
        held = str(d['held_out'])
        res = {'PICL unseen': unseen_source_readout(d, seed)[0], 'PICL in-distribution': d['te_P']}
        res.update({k: v[0] for k, v in baselines(d, seed).items()})
        for m, P in res.items():
            acc_P.setdefault((held, m), []).append(P)
        ys[held] = d['te_y']
    print(seed, flush=True)
methods = sorted({m for _, m in acc_P})
P_all = {m: np.vstack([np.mean(acc_P[(h, m)], 0) for h in sorted(ys)]) for m in methods}
y_all = np.concatenate([ys[h] for h in sorted(ys)])
N = len(y_all)
rng = np.random.default_rng(0)
rows = []
for m in methods:
    P = P_all[m]
    pred = P.argmax(1)
    corr = pred == y_all
    Pf = P @ AGG
    fcorr = Pf.argmax(1) == FAMILY[y_all]
    rows.append(
        dict(
            method=m, acc=corr.mean(), fam_acc=fcorr.mean(), aurc=aurc(P.max(1), corr), fam_aurc=aurc(Pf.max(1), fcorr)
        )
    )
summary = pd.DataFrame(rows).set_index('method')
print(summary.round(4).to_string())
ref = 'PICL unseen'
out = []
for m in methods:
    if m == ref:
        continue
    Pa, Pb = P_all[ref], P_all[m]
    ca, cb = Pa.argmax(1) == y_all, Pb.argmax(1) == y_all
    fa, fb = (Pa @ AGG).argmax(1) == FAMILY[y_all], (Pb @ AGG).argmax(1) == FAMILY[y_all]

    def mcnemar(x, y):
        b, c = int((x & ~y).sum()), int((~x & y).sum())
        return b, c, float(stats.binomtest(b, b + c, 0.5).pvalue) if b + c > 0 else 1.0

    d_aurc, d_faurc = [], []
    for _ in range(2000):
        idx = rng.integers(0, N, N)
        d_aurc.append(aurc(Pa.max(1)[idx], ca[idx]) - aurc(Pb.max(1)[idx], cb[idx]))
        d_faurc.append(aurc((Pa @ AGG).max(1)[idx], fa[idx]) - aurc((Pb @ AGG).max(1)[idx], fb[idx]))
    b1, c1, p1 = mcnemar(ca, cb)
    b2, c2, p2 = mcnemar(fa, fb)
    out.append(
        dict(
            vs=m,
            acc_diff=ca.mean() - cb.mean(),
            acc_mcnemar_p=p1,
            fam_acc_diff=fa.mean() - fb.mean(),
            fam_mcnemar_p=p2,
            aurc_diff=np.mean(d_aurc),
            aurc_ci=(np.quantile(d_aurc, 0.025), np.quantile(d_aurc, 0.975)),
            fam_aurc_diff=np.mean(d_faurc),
            fam_aurc_ci=(np.quantile(d_faurc, 0.025), np.quantile(d_faurc, 0.975)),
        )
    )
o = pd.DataFrame(out)
pd.set_option('display.width', 250)
print(o.round(4).to_string())
o.to_csv('tables/v6_loso_record_tests.csv', index=False)
summary.to_csv('tables/v6_loso_record_summary.csv')
