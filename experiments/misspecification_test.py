"""Controlled sweep: when does an interventional fit score add to classifier confidence?

python experiments/misspecification_test.py --reps 40 --out tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

N_G = 5
THETA_FIXED = 60.0


def aurc(conf, correct):
    o = np.argsort(-conf)
    c = correct[o].astype(float)
    k = np.arange(1, len(c) + 1)
    return float(np.mean(np.cumsum(1.0 - c) / k))


def _ed_unit(x):
    return x / (1.0 + np.clip(x, 0, None))


def _es_unit(x):
    return 0.5 * (1.0 + np.tanh(x))


def make_means(theta_deg, rho, rng):
    a = rng.normal(size=N_G)
    a /= np.linalg.norm(a)
    b = rng.normal(size=N_G)
    b -= (b @ a) * a
    b /= np.linalg.norm(b)
    t = np.radians(theta_deg)
    return a, rho * (np.cos(t) * a + np.sin(t) * b)


def run_cell(cov_ratio, tail_df, sigma, n, seed, w=(0.4, 0.3, 0.3)):
    """cov_ratio = sd(class 1) / sd(class 0). 1.0 is homoskedastic.
    tail_df = Student-t degrees of freedom; np.inf is Gaussian."""
    rng = np.random.default_rng(seed)
    mu0, mu1 = make_means(THETA_FIXED, 1.0, rng)
    M = np.vstack([mu0, mu1])
    s0 = sigma / np.sqrt(cov_ratio)
    s1 = sigma * np.sqrt(cov_ratio)
    scales = np.array([s0, s1])

    def gen(m):
        y = rng.integers(0, 2, m)
        if np.isinf(tail_df):
            e = rng.normal(size=(m, N_G))
        else:
            e = rng.standard_t(df=tail_df, size=(m, N_G)) / np.sqrt(tail_df / (tail_df - 2))
        return M[y] + scales[y][:, None] * e, y

    Xtr, ytr = gen(n)
    Xte, yte = gen(n)
    P = LogisticRegression(max_iter=2000).fit(Xtr, ytr).predict_proba(Xte)
    nrm = np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9
    Es = 1.0 - np.linalg.norm(Xte[:, None, :] - M[None, :, :], axis=2) / nrm
    Ed = np.repeat(np.linalg.norm(Xte, axis=1, keepdims=True) / nrm, 2, axis=1)
    comp = w[0] * P + w[1] * _ed_unit(Ed) + w[2] * _es_unit(Es)
    ac, am = aurc(P.max(1), P.argmax(1) == yte), aurc(comp.max(1), comp.argmax(1) == yte)
    return dict(
        cov_ratio=cov_ratio,
        tail_df=(0.0 if np.isinf(tail_df) else tail_df),
        seed=seed,
        aurc_conf=ac,
        aurc_comp=am,
        gain_pct=100.0 * (ac - am) / max(ac, 1e-12),
    )


def main(out_dir, n, sigma, reps):
    ratios = [1.0, 1.25, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 3.0, 4.0]
    tails = [np.inf, 3.0]
    rows = [
        run_cell(cr, td, sigma, n, seed=7000 + 13 * r + int(100 * cr))
        for td in tails
        for cr in ratios
        for r in range(reps)
    ]
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'misspecification_test.csv', index=False)

    print(f'\n=== Heteroskedasticity sweep, angle fixed at {THETA_FIXED:.0f} deg ' f'(n={n}/split, {reps} reps) ===')
    for td in tails:
        key = 0.0 if np.isinf(td) else td
        sub = df[df.tail_df == key]
        lbl = 'Gaussian tails' if key == 0.0 else f'Student-t (df={key:.0f}) tails'
        agg = (
            sub.groupby('cov_ratio')
            .agg(
                aurc_conf=('aurc_conf', 'mean'),
                aurc_comp=('aurc_comp', 'mean'),
                gain=('gain_pct', 'mean'),
                sd=('gain_pct', 'std'),
            )
            .reset_index()
        )
        print(f'\n--- {lbl} ---')
        print(agg.round(4).to_string(index=False))
        m = sub.groupby('cov_ratio').gain_pct.agg(['mean', 'std', 'count'])
        neg = m[m['mean'] < 0]
        pos = m[m['mean'] >= 0]
        if len(neg) and len(pos):
            lo, hi = neg.index.max(), pos.index.min()
            print(
                f'sign change bracketed by measured grid points: '
                f'{lo} ({m.loc[lo, "mean"]:+.2f}%) -> {hi} ({m.loc[hi, "mean"]:+.2f}%)'
            )
            for r_ in (lo, hi):
                t_, p_ = stats.ttest_1samp(sub[sub.cov_ratio == r_].gain_pct, 0.0)
                print(
                    f'   ratio {r_}: gain {m.loc[r_, "mean"]:+.2f}% '
                    f'+/- {m.loc[r_, "std"]:.2f}, t vs 0 = {t_:+.2f}, p = {p_:.3f}'
                )
        r = stats.spearmanr(sub.cov_ratio, sub.gain_pct)
        print(f'Spearman(cov_ratio, gain) = {r.statistic:+.3f}, p = {r.pvalue:.2e}')
        homo = sub[sub.cov_ratio == 1.0].gain_pct
        t, p = stats.ttest_1samp(homo, 0.0)
        print(f'homoskedastic cell: gain = {homo.mean():+.2f}% ' f'(t vs 0: {t:+.2f}, p = {p:.2e})')

    print(f'\nWritten to {out_dir}/misspecification_test.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=800)
    ap.add_argument('--sigma', type=float, default=0.6)
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    main(Path(a.out), a.n, a.sigma, a.reps)
