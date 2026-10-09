"""Transformer-level paired bootstrap for the leave-one-source-out results (predictions averaged over seeds).

PYTHONPATH=.:experiments/v4 python experiments/v6/loso_transformer_tests.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v4'))
from core import aurc, list_arrays, load
from eval_loso import baselines, unseen_source_readout
from picl.risk import AGG, FAMILY

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', type=int, nargs='+', default=[52, 53, 54, 55, 56])
ap.add_argument('--n-boot', type=int, default=4000)
ap.add_argument('--seed', type=int, default=17624)
ap.add_argument('--out', default='tables/v6_loso_transformer_tests.csv')
a = ap.parse_args()

GASES = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
SOURCES = {
    'NCEPR': ['NCEPR'],
    'IEC TC 10': ['IEC TC 10'],
    'IEEE DataPort': ['IEEE DataPort'],
    'Published cases': ['Published cases'],
    'NE Grid+Fujian': ['NE Grid', 'Fujian'],
}
SOURCES.update({k.replace(' ', '_'): v for k, v in list(SOURCES.items())})
prov = pd.read_csv('data/dga_provenance.csv').query("group == 'single_fault' and label_confirmed and fault_record")


def case_ids(d, held):
    """case_id of every held-out test record, matched on its measured log1p concentrations."""
    sub = prov[prov.source.isin(SOURCES[held])]
    lp = np.log1p(sub[GASES].to_numpy(float))
    obs = ~np.isnan(lp)
    te_lp, te_obs = d['te_lp'], d['te_obs'].astype(bool)
    assert len(sub) == len(te_lp), (held, len(sub), len(te_lp))
    out = []
    for i in range(len(te_lp)):
        hit = (obs == te_obs[i]).all(1) & np.all(np.where(obs, np.abs(lp - te_lp[i]), 0) < 1e-3, axis=1)
        idx = np.flatnonzero(hit)
        assert len(idx) >= 1, (held, i)
        out.append(sub.case_id.values[idx[0]])
    return np.array(out)


P_acc, ys, cases = {}, {}, {}
for seed in a.seeds:
    for fn in list_arrays('results/v4/loso', f'seed{seed}_*'):
        d = load(fn)
        held = str(d['held_out'])
        res = {'PICL unseen': unseen_source_readout(d, seed)[0], 'PICL in-distribution': d['te_P']}
        res.update({k: v[0] for k, v in baselines(d, seed).items()})
        for m, P in res.items():
            P_acc.setdefault((held, m), []).append(P)
        ys[held] = d['te_y']
        if held not in cases:
            cases[held] = case_ids(d, held)
    print('seed', seed, flush=True)
methods = sorted({m for _, m in P_acc})
H = sorted(ys)
P_all = {m: np.vstack([np.mean(P_acc[(h, m)], 0) for h in H]) for m in methods}
y = np.concatenate([ys[h] for h in H])
case = np.concatenate([cases[h] for h in H])
uc = np.unique(case)
groups = [np.flatnonzero(case == c) for c in uc]
print('records', len(y), 'transformers or cases', len(uc))
rng = np.random.default_rng(a.seed)
idx_sets = [np.concatenate([groups[i] for i in rng.integers(0, len(uc), len(uc))]) for _ in range(a.n_boot)]
ref = 'PICL unseen'
Pa = P_all[ref]
ca, fa = Pa.argmax(1) == y, (Pa @ AGG).argmax(1) == FAMILY[y]
sa, fsa = Pa.max(1), (Pa @ AGG).max(1)
rows = []
for m in methods:
    if m == ref:
        continue
    Pb = P_all[m]
    cb, fb = Pb.argmax(1) == y, (Pb @ AGG).argmax(1) == FAMILY[y]
    sb, fsb = Pb.max(1), (Pb @ AGG).max(1)
    d_acc = [ca[s].mean() - cb[s].mean() for s in idx_sets]
    d_fam = [fa[s].mean() - fb[s].mean() for s in idx_sets]
    d_aurc = [aurc(sa[s], ca[s]) - aurc(sb[s], cb[s]) for s in idx_sets]
    d_faurc = [aurc(fsa[s], fa[s]) - aurc(fsb[s], fb[s]) for s in idx_sets]
    q = lambda v, p: float(np.quantile(v, p))
    rows.append(
        dict(
            vs=m,
            acc_diff=ca.mean() - cb.mean(),
            acc_ci_lo=q(d_acc, 0.025),
            acc_ci_hi=q(d_acc, 0.975),
            fam_acc_diff=fa.mean() - fb.mean(),
            fam_acc_ci_lo=q(d_fam, 0.025),
            fam_acc_ci_hi=q(d_fam, 0.975),
            aurc_diff=aurc(sa, ca) - aurc(sb, cb),
            aurc_ci_lo=q(d_aurc, 0.025),
            aurc_ci_hi=q(d_aurc, 0.975),
            fam_aurc_diff=aurc(fsa, fa) - aurc(fsb, fb),
            fam_aurc_ci_lo=q(d_faurc, 0.025),
            fam_aurc_ci_hi=q(d_faurc, 0.975),
            n_cases=len(uc),
            n_records=len(y),
        )
    )
o = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(o.round(4).to_string())
o.to_csv(a.out, index=False)
