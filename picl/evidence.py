"""Interventional evidence log q(k | y) of the fitted SCM for each candidate fault, with unmeasured gases marginalised."""

from __future__ import annotations

import math

import numpy as np
import torch

_LOG_2PI = math.log(2.0 * math.pi)


@torch.no_grad()
def scm_class_loglik(y: torch.Tensor, observed: torch.Tensor, sources: torch.Tensor, graph, scm):
    """Return (l, d_obs): l is (N, K) float64 log-likelihoods, d_obs the number of measured gases."""
    nf = graph.n_faults
    W = (graph.final_hard_adjacency() * graph.weight_matrix()).detach().double()
    W_do = W.clone()
    W_do[:, :nf] = 0.0
    I = torch.eye(W.shape[0], dtype=W.dtype)
    T = torch.linalg.solve(I - W_do, I)
    means = scm.intervene_on_faults(W, torch.eye(nf, dtype=W.dtype), nf).double().numpy()
    yy = y.double().cpu().numpy()
    obs = observed.cpu().numpy().astype(bool)
    src = sources.cpu().numpy()
    N, K = yy.shape[0], nf
    out = np.zeros((N, K))
    for s in np.unique(src):
        sig2 = scm.noise_variance(int(s)).detach().double().clone()
        sig2[:nf] = 1e-9
        Sigma = (T.T @ torch.diag(sig2) @ T)[nf:, nf:].numpy()
        in_s = src == s
        for pat in np.unique(obs[in_s], axis=0):
            rows = np.where(in_s & (obs == pat).all(1))[0]
            o = pat.astype(bool)
            if not o.any():
                continue
            S = Sigma[np.ix_(o, o)] + 1e-6 * np.eye(int(o.sum()))
            L = np.linalg.cholesky(S)
            logdet = 2.0 * np.log(np.diag(L)).sum()
            idx = np.where(o)[0]
            for k in range(K):
                r = yy[np.ix_(rows, idx)] - means[k, o]
                z = np.linalg.solve(L, r.T)
                out[rows, k] = -0.5 * ((z * z).sum(0) + logdet + o.sum() * _LOG_2PI)
    return out, obs.sum(1)


def scm_log_posterior(y, observed, sources, graph, scm) -> np.ndarray:
    l, _ = scm_class_loglik(y, observed, sources, graph, scm)
    m = l.max(1, keepdims=True)
    return l - (m + np.log(np.exp(l - m).sum(1, keepdims=True)))


def dataset_log_posterior(ds, graph, scm) -> np.ndarray:
    return scm_log_posterior(ds.gas_values, ds.observed_gases(graph.n_faults), ds.source, graph, scm)
