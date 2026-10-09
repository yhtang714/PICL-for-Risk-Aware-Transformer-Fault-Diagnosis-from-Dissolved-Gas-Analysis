"""Local recalibration of a transferred model with k labelled records of the held-out source.

PYTHONPATH=.:experiments/v4 python experiments/v5/eval_fewshot.py --seeds 52 53 54 55 56 --out tables
"""

from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_softmax
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v4'))
warnings.filterwarnings('ignore')
from core import load, list_arrays, aurc, acc_at, K
from eval_loso import unseen_source_readout, baselines
from picl.risk import AGG, FAMILY

KS = (10, 20, 40)
R = 10
LAM = 1.0


def fit_bias(logp, y, lam=LAM):
    def nll(p):
        z = logp / np.exp(p[0]) + p[1:]
        return -log_softmax(z, 1)[np.arange(len(y)), y].mean() + lam * (p[1:] ** 2).sum() / len(y)

    r = minimize(nll, np.zeros(K + 1), method='L-BFGS-B', bounds=[(-2, 2)] + [(-3, 3)] * K)
    return float(np.exp(r.x[0])), r.x[1:]


def apply_bias(P, T, b):
    return np.exp(log_softmax(np.log(P + 1e-9) / T + b, 1))


def metr(P, y):
    pred = P.argmax(1)
    corr = pred == y
    Pf = P @ AGG
    fc = Pf.argmax(1) == FAMILY[y]
    return dict(
        acc=corr.mean(),
        macro_f1=f1_score(y, pred, average='macro', labels=list(range(K)), zero_division=0),
        aurc=aurc(P.max(1), corr),
        acc85=acc_at(P.max(1), corr),
        fam_acc=fc.mean(),
        fam_aurc=aurc(Pf.max(1), fc),
    )


def draws(y, k, rng):
    cls = np.unique(y)
    per = max(1, k // len(cls))
    idx = []
    for c in cls:
        cidx = np.where(y == c)[0]
        idx += list(rng.choice(cidx, min(per, len(cidx)), replace=False))
    return np.array(sorted(idx))


def main(seeds, out_dir, root='results/v4/loso'):
    rows = []
    for seed in seeds:
        for fn in list_arrays(root, f'seed{seed}_*'):
            d = load(fn)
            held = str(d['held_out'])
            yte = d['te_y']
            res = {
                'PICL, unseen-source read-out': unseen_source_readout(d, seed)[0],
                'PICL in-distribution read-out': d['te_P'],
            }
            res.update({k: v[0] for k, v in baselines(d, seed).items()})
            rng = np.random.default_rng(1000 + seed)
            plan = {(k, r): draws(yte, k, rng) for k in KS for r in range(R)}
            for name, P in res.items():
                rows.append(dict(seed=seed, held_out=held, n=len(yte), method=name, k=0, rep=0, **metr(P, yte)))
                for (k, r), idx in plan.items():
                    rest = np.setdiff1d(np.arange(len(yte)), idx)
                    T_, b_ = fit_bias(np.log(P[idx] + 1e-9), yte[idx])
                    rows.append(
                        dict(
                            seed=seed,
                            held_out=held,
                            n=len(rest),
                            method=name,
                            k=k,
                            rep=r,
                            n_labelled=len(idx),
                            **metr(apply_bias(P, T_, b_)[rest], yte[rest]),
                            **{f'{kk}_noadapt': v for kk, v in metr(P[rest], yte[rest]).items()},
                        )
                    )
            print(seed, held, flush=True)
        pd.DataFrame(rows).to_csv(out_dir / 'v5_fewshot_per_fold.csv', index=False)
    d = pd.DataFrame(rows)
    M = ['acc', 'macro_f1', 'aurc', 'acc85', 'fam_acc', 'fam_aurc']
    out = []
    for k in (0,) + KS:
        x = d[d.k == k]
        cols = M + ([f'{c}_noadapt' for c in M] if k > 0 else [])
        mic = (
            x.groupby(['seed', 'rep', 'method'])
            .apply(lambda g: pd.Series({c: np.average(g[c], weights=g['n']) for c in cols}), include_groups=False)
            .reset_index()
        )
        mic['k'] = k
        out.append(mic)
    mic = pd.concat(out, ignore_index=True)
    mic.to_csv(out_dir / 'v5_fewshot_micro_per_seed_rep.csv', index=False)
    agg = mic.groupby(['k', 'method'])[M].agg(['mean', 'std'])
    agg.columns = [f'{a}_{b}' for a, b in agg.columns]
    agg.reset_index().to_csv(out_dir / 'v5_fewshot_summary.csv', index=False)
    pd.set_option('display.width', 250)
    for k in (0,) + KS:
        print(f'=== k = {k} ===')
        print(mic[mic.k == k].groupby('method')[M].mean().round(4).to_string())
    ps = mic.groupby(['k', 'seed', 'method'])[M].mean().reset_index()
    for k in (0,) + KS:
        a = ps[(ps.k == k) & (ps.method == 'PICL, unseen-source read-out')].set_index('seed')
        for m in ps.method.unique():
            if m.startswith('PICL, unseen'):
                continue
            b = ps[(ps.k == k) & (ps.method == m)].set_index('seed')
            print(
                k,
                m,
                {
                    c: (
                        round(float((a[c] - b[c]).mean()), 3),
                        (
                            int(((a[c] - b[c]) > 0).sum())
                            if c not in ('aurc', 'fam_aurc')
                            else int(((a[c] - b[c]) < 0).sum())
                        ),
                    )
                    for c in ('acc', 'macro_f1', 'aurc', 'fam_acc', 'fam_aurc')
                },
            )


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[52, 53, 54, 55, 56])
    ap.add_argument('--out', default='tables')
    ap.add_argument('--root', default='results/v4/loso')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    main(a.seeds, out, a.root)
