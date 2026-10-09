"""Decision layer and unseen-source read-out: scale-free evidence, two-fault screen and hierarchical decisions."""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

NF, G, K = 6, 5, 6
LOG2PI = math.log(2 * math.pi)
PAIRS_GAS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
PAIRS_CLASS = list(itertools.combinations(range(K), 2))
FAMILY = np.array([0, 1, 1, 2, 2, 2])
FAMILY_NAMES = ['PD', 'discharge', 'thermal']
AGG = np.zeros((K, 3))
AGG[np.arange(K), FAMILY] = 1.0


class SCMArrays:
    """Interventional means and noise covariances of a fitted SCM (numpy)."""

    def __init__(self, W, mu, log_sigma2):
        self.mu, self.ls2 = np.asarray(mu, float), np.asarray(log_sigma2, float)
        Wdo = np.array(W, float)
        Wdo[:, :NF] = 0.0
        I = np.eye(Wdo.shape[0])
        self.T = np.linalg.solve(I - Wdo, I)
        Tfy = self.T[:NF, NF:]
        mu_f, mu_y = self.mu[:NF], self.mu[NF:]
        self.m0 = (0 - mu_f) @ Tfy + mu_y
        self.means = (np.eye(NF) - mu_f) @ Tfy + mu_y
        self.delta = self.means - self.m0
        self.n_src = self.ls2.shape[0]

    @classmethod
    def from_model(cls, graph, scm):
        import torch

        with torch.no_grad():
            W = (graph.final_hard_adjacency() * graph.weight_matrix()).detach().double().numpy()
            return cls(W, scm.mu.detach().double().numpy(), scm.log_sigma2.detach().double().numpy())

    def sigma(self, s):
        sig2 = np.exp(self.ls2.mean(0)) if (s < 0 or s >= self.n_src) else np.exp(self.ls2[s])
        sig2 = np.maximum(sig2, 1e-6).copy()
        sig2[:NF] = 1e-9
        return (self.T.T @ np.diag(sig2) @ self.T)[NF:, NF:]


def gaussian_loglik(Y, obs, src, scm: SCMArrays, means, extra_cov=None):
    """(N, H) log N(y_o; means[h]_o, Sigma(src)_oo + extra_cov[h]_oo) on the measured gases."""
    Y, obs, src = np.asarray(Y, float), np.asarray(obs, bool), np.asarray(src)
    out = np.zeros((len(Y), len(means)))
    for s in np.unique(src):
        S = scm.sigma(int(s))
        ins = src == s
        for pat in np.unique(obs[ins], axis=0):
            rows = np.where(ins & (obs == pat).all(1))[0]
            o = pat.astype(bool)
            if not o.any():
                continue
            for h in range(len(means)):
                C = S[np.ix_(o, o)] + 1e-6 * np.eye(o.sum())
                if extra_cov is not None:
                    C = C + extra_cov[h][np.ix_(o, o)]
                L = np.linalg.cholesky(C)
                z = np.linalg.solve(L, (Y[np.ix_(rows, np.where(o)[0])] - means[h, o]).T)
                out[rows, h] = -0.5 * ((z * z).sum(0) + 2 * np.log(np.diag(L)).sum() + o.sum() * LOG2PI)
    return out


def log_posterior(ll):
    return ll - logsumexp(ll, axis=1, keepdims=True)


class ScaleFreeEvidence:
    def __init__(self, scm: SCMArrays, log_sd):
        self.scm, self.u = scm, 1.0 / np.asarray(log_sd, float)
        self.tau, self.tau_s = np.zeros(K), 0.0

    def _extra(self, tau, tau_s):
        E = np.stack([tau[k] ** 2 * np.outer(self.scm.delta[k], self.scm.delta[k]) for k in range(K)])
        return E + tau_s**2 * np.outer(self.u, self.u)[None]

    def loglik(self, Y, obs, src):
        return gaussian_loglik(Y, obs, src, self.scm, self.scm.means, self._extra(self.tau, self.tau_s))

    def fit(self, Y, obs, src, y):
        y = np.asarray(y)

        def nll(p):
            ll = gaussian_loglik(Y, obs, src, self.scm, self.scm.means, self._extra(np.exp(p[:K]), np.exp(p[K])))
            return -ll[np.arange(len(y)), y].mean()

        r = minimize(
            nll, np.full(K + 1, np.log(0.5)), method='L-BFGS-B', bounds=[(-5, 2)] * (K + 1), options=dict(maxiter=200)
        )
        self.tau, self.tau_s = np.exp(r.x[:K]), float(np.exp(r.x[K]))
        return self

    def log_q(self, Y, obs, src):
        return log_posterior(self.loglik(Y, obs, src))


def log_ratios(lp):
    lp = np.asarray(lp, float)
    return np.column_stack([lp[:, i] - lp[:, j] for i, j in PAIRS_GAS])


def readout_features(gas_z, lp, logq, with_levels=True):
    parts = ([np.asarray(gas_z, float)] if with_levels else []) + [log_ratios(lp), np.asarray(logq, float)]
    return np.hstack(parts).astype(np.float32)


def proba_k(model, X):
    P = np.zeros((len(X), K))
    P[:, np.asarray(model.classes_).astype(int)] = model.predict_proba(X)
    return P


def class_source_weights(y, src):
    """Class-balanced weights rescaled so that every training source carries the same total weight."""
    from sklearn.utils.class_weight import compute_sample_weight

    w = compute_sample_weight('balanced', y)
    srcs = np.unique(src)
    for s in srcs:
        m = src == s
        w[m] *= len(y) / (len(srcs) * m.sum())
    return w / w.mean()


def ppm_mixtures(lp, y, src, rng, lams, n_pairs):
    """ppm-additive mixtures of records of two different classes: lam * ppm_a + (1 - lam) * ppm_b."""
    ppm = np.expm1(np.asarray(lp, float))
    out, pa, pb, so, lo = [], [], [], [], []
    for a, c in PAIRS_CLASS:
        A, B = np.where(y == a)[0], np.where(y == c)[0]
        if len(A) == 0 or len(B) == 0:
            continue
        for lam in lams:
            ia, ib = rng.choice(A, n_pairs), rng.choice(B, n_pairs)
            out.append(np.log1p(lam * ppm[ia] + (1 - lam) * ppm[ib]))
            pa += [a] * n_pairs
            pb += [c] * n_pairs
            so.append(src[ia])
            lo += [lam] * n_pairs
    return np.vstack(out), np.array(pa), np.array(pb), np.concatenate(so), np.array(lo)


TRAIN_LAMS = tuple(np.round(np.linspace(0.1, 0.9, 9), 2))


class TwoFaultScreen:
    """Detector of records that resemble a simultaneous two-fault record."""

    def __init__(self, seed, n_pairs=15, flag_rate=0.10):
        self.seed, self.n_pairs, self.flag_rate = seed, n_pairs, flag_rate
        self.threshold = np.inf

    def fit(self, X_real, lp_complete, y_complete, src_complete, feature_fn, exclude_pair=None):
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.default_rng(self.seed)
        lp_m, pa, pb, s_m, _ = ppm_mixtures(lp_complete, y_complete, src_complete, rng, TRAIN_LAMS, self.n_pairs)
        if exclude_pair is not None:
            keep = ~(
                ((pa == exclude_pair[0]) & (pb == exclude_pair[1]))
                | ((pa == exclude_pair[1]) & (pb == exclude_pair[0]))
            )
            lp_m, s_m = lp_m[keep], s_m[keep]
        X_mix = feature_fn(lp_m, s_m)
        X = np.vstack([X_real, X_mix])
        yy = np.r_[np.zeros(len(X_real)), np.ones(len(X_mix))]
        self.model = RandomForestClassifier(
            n_estimators=500, random_state=self.seed, n_jobs=2, class_weight='balanced', min_samples_leaf=2
        ).fit(X, yy)
        return self

    def score(self, X):
        for_est = self.model
        if hasattr(for_est, 'n_jobs'):
            for_est.n_jobs = 1
        return for_est.predict_proba(X)[:, 1]

    def set_threshold(self, X_cal):
        self.threshold = float(np.quantile(self.score(X_cal), 1 - self.flag_rate))
        return self

    def flag(self, X):
        return self.score(X) > self.threshold


def hierarchical_thresholds(P_cal, cov_subclass=0.70, cov_total=0.95, flag_cal=None):
    flag = np.zeros(len(P_cal), bool) if flag_cal is None else np.asarray(flag_cal, bool)
    c = np.where(flag, -1.0, P_cal.max(1))
    t_sub = float(np.quantile(c, 1 - cov_subclass, method='lower'))
    rest = c < t_sub
    need = cov_total - (1 - rest.mean())
    if need <= 0 or not rest.any():
        return t_sub, np.inf
    cf = np.where(flag[rest], -1.0, (P_cal[rest] @ AGG).max(1))
    t_fam = float(np.quantile(cf, min(max(1 - need / rest.mean(), 0.0), 1.0), method='lower'))
    return t_sub, t_fam


def hierarchical_decide(P, t_sub, t_fam, flag=None):
    """level 2 = sub-class, 1 = family, 0 = referred."""
    flag = np.zeros(len(P), bool) if flag is None else np.asarray(flag, bool)
    sub = (~flag) & (P.max(1) >= t_sub)
    fam = (~flag) & (~sub) & ((P @ AGG).max(1) >= t_fam)
    return np.where(sub, 2, np.where(fam, 1, 0)), P.argmax(1), (P @ AGG).argmax(1)
