"""Evidence variants of the SCM (fault-dependent noise, Student-t residuals) with the read-out kept fixed.

python experiments/picl_v3_explore.py --seeds 52 ... 61 --out tables
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import gammaln, logsumexp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import TARGET_COVERAGE, accepted_acc_at_coverage, aurc, load_seed
from picl.classifier_head import _dga_ratios
from picl_v2_explore import fit_T_beta, fused

warnings.filterwarnings('ignore')
K, NF = 6, 6
LABELS = list(range(K))
_LOG2PI = math.log(2 * math.pi)


def scm_parts(b):
    W = b.W_eff.double()
    W_do = W.clone()
    W_do[:, :NF] = 0.0
    I = torch.eye(W.shape[0], dtype=W.dtype)
    T = torch.linalg.solve(I - W_do, I)
    with torch.no_grad():
        means = b.scm.intervene_on_faults(W, torch.eye(NF, dtype=W.dtype), NF).double().numpy()
        mu = b.scm.mu.detach().double().numpy()
        sig2 = {s: b.scm.noise_variance(int(s)).detach().double().numpy() for s in range(-1, b.scm.n_sources)}
    return W.numpy(), T.numpy(), means, mu, sig2


def structural_residuals(b, raw, W, mu):
    """e = x_c (I - W) on the gas columns for records with all gases observed, faults at the label."""
    y = raw.gas_values.double().numpy()
    miss = raw.miss_mask[:, NF:].numpy().astype(bool)
    lab = raw.labels.numpy()
    keep = ~miss.any(1)
    x = np.hstack([np.eye(NF)[lab], y])[keep]
    xc = x - mu
    e = (xc @ (np.eye(W.shape[0]) - W))[:, NF:]
    return e, lab[keep], raw.source.numpy()[keep]


def fault_noise_scales(b, n0=10.0):
    """s_jk: ratio of the class-k structural residual variance of gas j to the source noise."""
    W, T, means, mu, sig2 = scm_parts(b)
    e, lab, src = structural_residuals(b, b.train_raw, W, mu)
    ratio = e**2 / np.stack([sig2[int(s)][NF:] for s in src])
    S = np.ones((K, 5))
    for k in range(K):
        r = ratio[lab == k]
        S[k] = (r.sum(0) + n0) / (len(r) + n0)
    return S


def class_residual_cov(b):
    """Empirical covariance of y - m_k over complete training records of class k (pooled over sources)."""
    W, T, means, mu, sig2 = scm_parts(b)
    raw = b.train_raw
    y = raw.gas_values.double().numpy()
    miss = raw.miss_mask[:, NF:].numpy().astype(bool)
    lab = raw.labels.numpy()
    keep = ~miss.any(1)
    out, n = [], []
    for k in range(K):
        r = y[keep & (lab == k)] - means[k]
        out.append(r.T @ r / max(len(r), 1))
        n.append(len(r))
    return np.array(out), np.array(n)


def loglik(b, raw, variant, S=None, Ck=None, nk=None, nu=5.0, n0_cov=20.0):
    W, T, means, mu, sig2 = scm_parts(b)
    y = raw.gas_values.double().numpy()
    miss = raw.miss_mask[:, NF:].numpy().astype(bool)
    src = raw.source.numpy()
    N = len(y)
    out = np.zeros((N, K))
    student = variant in ('E2', 'E3')
    for s in np.unique(src):
        base = sig2[int(s)] if int(s) in sig2 else sig2[-1]
        covs = []
        for k in range(K):
            d = base.copy()
            d[:NF] = 1e-9
            if variant in ('E1', 'E2'):
                d[NF:] = d[NF:] * S[k]
            Sig = (T.T @ np.diag(d) @ T)[NF:, NF:]
            if variant == 'E4':
                lam = nk[k] / (nk[k] + n0_cov)
                Sig = (1 - lam) * Sig + lam * Ck[k]
            covs.append(Sig)
        for pat in np.unique(miss[src == s], axis=0):
            rows = np.where((src == s) & (miss == pat).all(1))[0]
            o = ~pat
            if not o.any():
                continue
            d_o = int(o.sum())
            for k in range(K):
                Sk = covs[k][np.ix_(o, o)] + 1e-6 * np.eye(d_o)
                if student:
                    Sk = Sk * (nu - 2) / nu
                L = np.linalg.cholesky(Sk)
                logdet = 2 * np.log(np.diag(L)).sum()
                r = y[np.ix_(rows, np.where(o)[0])] - means[k, o]
                z = np.linalg.solve(L, r.T)
                q = (z * z).sum(0)
                if student:
                    out[rows, k] = (
                        gammaln((nu + d_o) / 2)
                        - gammaln(nu / 2)
                        - 0.5 * d_o * np.log(nu * np.pi)
                        - 0.5 * logdet
                        - 0.5 * (nu + d_o) * np.log1p(q / nu)
                    )
                else:
                    out[rows, k] = -0.5 * (q + logdet + d_o * _LOG2PI)
    return out


def rf(seed):
    return RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=2, class_weight='balanced')


def proba(m, X):
    P = np.zeros((len(X), K))
    P[:, m.classes_.astype(int)] = m.predict_proba(X)
    return P


def metrics(P, y, Pcal=None):
    pred = P.argmax(1)
    corr = pred == y
    conf = P.max(1)
    Pc = np.clip(P, 1e-9, 1)
    Pc = Pc / Pc.sum(1, keepdims=True)
    return dict(
        acc=float(corr.mean()),
        bal_acc=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average='macro', labels=LABELS, zero_division=0)),
        aurc=aurc(conf, corr),
        acc85=accepted_acc_at_coverage(conf, corr, TARGET_COVERAGE)['accepted_acc'],
        nll=float(-np.log(Pc[np.arange(len(y)), y]).mean()),
        brier=float(((P - np.eye(K)[y]) ** 2).sum(1).mean()),
    )


def readout(b, feats, seed):
    """RF on [gas, ratios, logq] + fusion; returns test probabilities and (T, beta)."""
    ytr, yca = b.train.labels.numpy(), b.cal.labels.numpy()
    X = {n: np.hstack([f['gr'], f['logq']]) for n, f in feats.items()}
    m = rf(seed).fit(X['tr'], ytr)
    lPc, lPt = np.log(proba(m, X['ca']) + 1e-6), np.log(proba(m, X['te']) + 1e-6)
    T, beta = fit_T_beta(lPc, feats['ca']['logq'], yca, fuse=True)
    return fused(lPt, feats['te']['logq'], T, beta), fused(lPc, feats['ca']['logq'], T, beta), T, beta


def base_feats(b):
    out = {}
    for n, ds in (('tr', b.train), ('ca', b.cal), ('te', b.test)):
        out[n] = dict(gr=np.hstack([ds.gas_values.numpy(), _dga_ratios(ds.log_ppm).numpy()]))
    return out


VARIANTS = ['E0', 'E1', 'E2', 'E3', 'E4']


def main(seeds, out_dir):
    rows = []
    for seed in seeds:
        b = load_seed(seed)
        yte = b.test.labels.numpy()
        S = fault_noise_scales(b)
        Ck, nk = class_residual_cov(b)
        bf = base_feats(b)
        for v in VARIANTS:
            feats = {}
            for n, raw in (('tr', b.train_raw), ('ca', b.cal_raw), ('te', b.test_raw)):
                ll = loglik(b, raw, v, S=S, Ck=Ck, nk=nk)
                feats[n] = dict(gr=bf[n]['gr'], logq=ll - logsumexp(ll, axis=1, keepdims=True))
            Pt, Pc, T, beta = readout(b, feats, seed)
            ev = feats['te']['logq'].argmax(1) == yte
            rows.append(dict(seed=seed, variant=v, T=T, beta=beta, evidence_alone=float(ev.mean()), **metrics(Pt, yte)))
            print(
                seed, v, {k: round(val, 3) for k, val in rows[-1].items() if k not in ('seed', 'variant')}, flush=True
            )
        rows.append(dict(seed=seed, variant='S_jk', **{f's_{k}_{j}': S[k, j] for k in range(K) for j in range(5)}))
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'picl_v3_evidence_per_seed.csv', index=False)
    d = df[df.variant != 'S_jk']
    agg = d.groupby('variant')[
        ['acc', 'bal_acc', 'macro_f1', 'aurc', 'acc85', 'nll', 'brier', 'evidence_alone', 'beta']
    ].mean()
    print(agg.round(4).to_string())
    base = d[d.variant == 'E0'].set_index('seed')
    for v in VARIANTS[1:]:
        o = d[d.variant == v].set_index('seed')
        dd = o[['acc', 'macro_f1', 'aurc', 'acc85', 'nll']] - base[['acc', 'macro_f1', 'aurc', 'acc85', 'nll']]
        print(
            v,
            'mean diff vs E0:',
            dd.mean().round(4).to_dict(),
            'wins(aurc lower):',
            int((dd.aurc < 0).sum()),
            'wins(acc):',
            int((dd.acc > 0).sum()),
        )
    sj = df[df.variant == 'S_jk'].drop(columns=['variant', 'seed']).dropna(axis=1).mean()
    print(np.array(sj).reshape(K, 5).round(2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 52)))
    ap.add_argument('--out', default='tables/design')
    a = ap.parse_args()
    main(a.seeds, Path(a.out))
