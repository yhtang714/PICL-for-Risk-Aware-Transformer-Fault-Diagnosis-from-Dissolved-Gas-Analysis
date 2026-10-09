"""Array-level implementation of the PICL read-out used by the design and evaluation scripts."""

from __future__ import annotations

import io
import zipfile
from fnmatch import fnmatch
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, logsumexp
from sklearn.metrics import f1_score

NF, G, K = 6, 5, 6
PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
LOG2PI = math.log(2 * math.pi)


ARRAYS = Path('results/arrays.zip')


def _member(fn):
    p = Path(fn).as_posix()
    i = p.find('results/')
    return p[i + len('results/'):] if i >= 0 else p


def load(fn):
    if Path(fn).exists():
        z = np.load(fn, allow_pickle=True)
    else:
        with zipfile.ZipFile(ARRAYS) as zf:
            z = np.load(io.BytesIO(zf.read(_member(fn))), allow_pickle=True)
    return {k: z[k] for k in z.files}


def list_arrays(root, pattern):
    root = Path(root)
    found = {f.name: f for f in root.glob(pattern + '.npz')}
    if ARRAYS.exists():
        pre = _member(root).rstrip('/') + '/'
        with zipfile.ZipFile(ARRAYS) as zf:
            for n in zf.namelist():
                name = n[len(pre):]
                if n.startswith(pre) and '/' not in name and fnmatch(name, pattern + '.npz'):
                    found.setdefault(name, root / name)
    return [found[k] for k in sorted(found)]


class SCM:
    def __init__(self, d):
        W = d['W']
        self.mu = d['mu']
        self.ls2 = d['log_sigma2']
        Wdo = W.copy()
        Wdo[:, :NF] = 0.0
        I = np.eye(W.shape[0])
        self.T = np.linalg.solve(I - Wdo, I)
        Tfy = self.T[:NF, NF:]
        mu_f, mu_y = self.mu[:NF], self.mu[NF:]
        self.m0 = (0 - mu_f) @ Tfy + mu_y
        self.means = (np.eye(NF) - mu_f) @ Tfy + mu_y
        self.delta = self.means - self.m0
        self.n_src = self.ls2.shape[0]

    def sigma(self, s):
        if s < 0 or s >= self.n_src:
            sig2 = np.exp(self.ls2.mean(0))
        else:
            sig2 = np.exp(self.ls2[s])
        sig2 = np.maximum(sig2, 1e-6).copy()
        sig2[:NF] = 1e-9
        return (self.T.T @ np.diag(sig2) @ self.T)[NF:, NF:]


def gauss_ll(Y, obs, src, scm, means, extra_cov=None):
    """log N(y_o; means[k]_o, Sigma(src)_oo + extra_cov[k]_oo) for every row and hypothesis.
    means: (H, G); extra_cov: None or (H, G, G)."""
    N, H = Y.shape[0], means.shape[0]
    out = np.zeros((N, H))
    for s in np.unique(src):
        S = scm.sigma(int(s))
        ins = src == s
        for pat in np.unique(obs[ins], axis=0):
            rows = np.where(ins & (obs == pat).all(1))[0]
            o = pat.astype(bool)
            if not o.any():
                continue
            for h in range(H):
                C = S[np.ix_(o, o)] + 1e-6 * np.eye(o.sum())
                if extra_cov is not None:
                    C = C + extra_cov[h][np.ix_(o, o)]
                L = np.linalg.cholesky(C)
                r = Y[np.ix_(rows, np.where(o)[0])] - means[h, o]
                zz = np.linalg.solve(L, r.T)
                out[rows, h] = -0.5 * ((zz * zz).sum(0) + 2 * np.log(np.diag(L)).sum() + o.sum() * LOG2PI)
    return out


def logq_from_ll(ll):
    return ll - logsumexp(ll, axis=1, keepdims=True)


def ratios(lp, obs=None):
    R = np.column_stack([lp[:, i] - lp[:, j] for i, j in PAIRS])
    return R


def head_X(gas, lp, lq):
    return np.hstack([gas, ratios(lp), lq]).astype(np.float32)


def base_X(gas, lp):
    return np.hstack([gas, ratios(lp)]).astype(np.float32)


def fit_T_beta(P, lq, y, fuse=True):
    lp = np.log(np.asarray(P) + 1e-6)

    def nll(p):
        z = lp / np.exp(p[0]) + (p[1] if fuse else 0.0) * lq
        return -log_softmax(z, axis=1)[np.arange(len(y)), y].mean()

    best = None
    for b0 in ([0.0, 0.3, 1.0] if fuse else [0.0]):
        r = minimize(nll, x0=[0.0, b0], method='L-BFGS-B', bounds=[(-3, 3), (0, 3) if fuse else (0, 0)])
        if best is None or r.fun < best.fun:
            best = r
    return float(np.exp(best.x[0])), float(best.x[1]) if fuse else 0.0


def fused(P, lq, T, beta):
    z = np.log(np.asarray(P) + 1e-6) / T + beta * lq
    return np.exp(log_softmax(z, axis=1))


def temp_fit(P, y):
    T, _ = fit_T_beta(P, np.zeros_like(P), y, fuse=False)
    return T


def temp_apply(P, T):
    return np.exp(log_softmax(np.log(np.asarray(P) + 1e-6) / T, axis=1))


def proba6(m, X):
    P = np.zeros((len(X), K))
    P[:, np.asarray(m.classes_).astype(int)] = m.predict_proba(X)
    return P


def aurc(conf, correct):
    o = np.argsort(-conf, kind='stable')
    c = correct[o].astype(float)
    return float(np.mean(np.cumsum(1 - c) / np.arange(1, len(c) + 1)))


def acc_at(conf, correct, cov=0.85):
    k = int(round(cov * len(conf)))
    o = np.argsort(-conf, kind='stable')[:k]
    return float(correct[o].mean())


def thr_for(conf_cal, cov=0.85):
    return float(np.quantile(conf_cal, 1 - cov, method='lower'))


def scores(P, y, conf=None, P_cal=None, y_cal=None, conf_cal=None):
    pred = P.argmax(1)
    corr = pred == y
    conf = P.max(1) if conf is None else conf
    d = dict(
        acc=float(corr.mean()),
        f1=float(f1_score(y, pred, average='macro', labels=list(range(K)), zero_division=0)),
        aurc=aurc(conf, corr),
        acc85=acc_at(conf, corr),
    )
    if P_cal is not None:
        cc = P_cal.max(1) if conf_cal is None else conf_cal
        t = thr_for(cc)
        m = conf >= t
        d['cov_tr'] = float(m.mean())
        d['acc_tr'] = float(corr[m].mean()) if m.any() else np.nan
    return d


def nb_test(diff, n_train, n_test):
    """Nadeau-Bengio corrected resampled t-test; returns (mean, p)."""
    from scipy import stats

    diff = np.asarray(diff, float)
    J = len(diff)
    m, v = diff.mean(), diff.var(ddof=1)
    se = math.sqrt((1 / J + n_test / n_train) * v) if v > 0 else 1e-12
    t = m / se
    return float(m), float(2 * stats.t.sf(abs(t), J - 1))
