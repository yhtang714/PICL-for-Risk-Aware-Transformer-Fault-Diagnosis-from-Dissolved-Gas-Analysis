from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn

_LOG_2PI = math.log(2.0 * math.pi)


def _class_means(W_eff, n_faults):
    n_vars = W_eff.shape[0]
    W_int = W_eff.clone()
    W_int[:, :n_faults] = 0.0
    I = torch.eye(n_vars, dtype=W_eff.dtype, device=W_eff.device)
    T = torch.linalg.solve(I - W_int, I)
    E_f = torch.eye(n_faults, n_vars, dtype=W_eff.dtype, device=W_eff.device)
    return (E_f @ T)[:, n_faults:]


def class_posterior_single_graph(y, W_eff, sigma2, mu, n_faults):
    n_vars = W_eff.shape[0]
    I = torch.eye(n_vars, dtype=W_eff.dtype, device=W_eff.device)
    T = torch.linalg.solve(I - W_eff, I)
    Sigma = T.T @ torch.diag(sigma2) @ T + 1e-06 * torch.eye(n_vars)
    Omega = torch.linalg.inv(Sigma)
    Omega_ff = Omega[:n_faults, :n_faults]
    Omega_fy = Omega[:n_faults, n_faults:]
    mu_f = mu[:n_faults]
    mu_y = mu[n_faults:]
    I_K = torch.eye(n_faults, dtype=W_eff.dtype, device=W_eff.device)
    centred_f = I_K - mu_f.unsqueeze(0)
    centred_y = y - mu_y.unsqueeze(0)
    quad_f = (centred_f @ Omega_ff * centred_f).sum(dim=1)
    M = centred_f @ Omega_fy
    cross = centred_y @ M.T
    log_joint = -0.5 * quad_f.unsqueeze(0) - cross
    return log_joint - torch.logsumexp(log_joint, dim=1, keepdim=True)


def class_posterior_bma(y, graph, scm, n_samples, source_idx=0, tau=0.05):
    n_faults = graph.n_faults
    sigma2 = scm.noise_variance(source_idx).detach()
    mu = scm.mu.detach()
    log_posts = []
    with torch.no_grad():
        for _ in range(n_samples):
            A = graph.sample_adjacency(tau=tau, hard=False)
            W_eff = A * graph.weight_matrix()
            log_posts.append(class_posterior_single_graph(y, W_eff.detach(), sigma2, mu, n_faults))
    stacked = torch.stack(log_posts, dim=0)
    return stacked.exp().mean(dim=0).clamp(min=1e-12)


def causal_disablement_and_sufficiency(y, W_eff, scm, n_faults, activations=None, clamp=False, obs=None):
    """Disablement E_d and sufficiency E_s(f_k) under single-fault interventions."""
    B, G = y.shape
    K = n_faults
    dtype, device = (y.dtype, y.device)
    om = None if obs is None else obs.to(dtype=dtype, device=device)
    _nrm = (
        (lambda r: torch.norm(r, dim=-1))
        if om is None
        else (lambda r: torch.norm(r * (om if r.dim() == 2 else om.unsqueeze(1)), dim=-1))
    )
    y_norm = _nrm(y).unsqueeze(1).clamp(min=1e-08)

    if activations is None:
        act = torch.ones(B, K, dtype=dtype, device=device)
    else:
        act = activations.to(dtype=dtype, device=device)
    f_es = torch.zeros(B, K, K, dtype=dtype, device=device)
    for k in range(K):
        f_es[:, k, k] = act[:, k]
    mu_es = scm.intervene_on_faults(W_eff, f_es.reshape(B * K, K), n_faults).reshape(B, K, G)
    Es = 1.0 - _nrm(y.unsqueeze(1) - mu_es) / y_norm

    if activations is None:
        f_zero = torch.zeros(B, K, dtype=dtype, device=device)
        mu_0 = scm.intervene_on_faults(W_eff, f_zero, n_faults)
        Ed = (_nrm(y - mu_0).unsqueeze(1) / y_norm).expand(B, K).clone()
    else:
        f_ed = act.unsqueeze(1).expand(B, K, K).clone()
        for k in range(K):
            f_ed[:, k, k] = 0.0
        mu_ed = scm.intervene_on_faults(W_eff, f_ed.reshape(B * K, K), n_faults).reshape(B, K, G)
        Ed = _nrm(y.unsqueeze(1) - mu_ed) / y_norm

    if clamp:
        Es = Es.clamp(0.0, 1.0)
        Ed = Ed.clamp(0.0, 10.0)
    return (Ed, Es)


def _gate_activations(cfg, y, graph, scm, sources, n_graph_samples, classifier_posterior):
    """Resolve the activation pattern for E_d / E_s from the config."""
    mode = str(cfg.raw['inference'].get('intervention_source', 'onehot'))
    if mode == 'onehot':
        return None
    if mode == 'scm_post':
        B = y.shape[0]
        post = torch.zeros(B, graph.n_faults, dtype=y.dtype, device=y.device)
        for s in torch.unique(sources):
            sel = sources == s
            post[sel] = class_posterior_bma(y[sel], graph, scm, n_graph_samples, source_idx=int(s.item()))
        return post
    if mode == 'classifier':
        if classifier_posterior is None:
            raise ValueError("intervention_source='classifier' needs a classifier posterior")
        return classifier_posterior.to(y.device)
    raise ValueError(f'unknown intervention_source: {mode}')


def _ed_unit(Ed):
    """Map E_d from [0, inf) to [0, 1) with x / (1 + x)."""
    return Ed / (1.0 + Ed.clamp(min=0.0))


def _es_unit(Es):
    """Map E_s in (-inf, 1] onto (0, 1) monotonically:  (1 + tanh(x)) / 2."""
    return 0.5 * (1.0 + torch.tanh(Es))


@dataclass
class CompositeScores:
    posteriors: torch.Tensor
    Ed: torch.Tensor
    Es: torch.Tensor
    scores: torch.Tensor
    pred: torch.Tensor
    conf: torch.Tensor


def composite_scores(cfg, y, graph, scm, n_graph_samples, sources=None, classifier_posterior=None):
    """Composite deferral score from the class probabilities and the disablement and sufficiency scores."""
    w = cfg.raw['inference']['gate_weights']
    wp, wd, ws = (float(w['posterior']), float(w['disablement']), float(w['sufficiency']))
    clamp = bool(cfg.raw['inference'].get('clamp_scores', False))
    B = y.shape[0]
    if sources is None:
        sources = torch.zeros(B, dtype=torch.long, device=y.device)
    with torch.no_grad():
        if classifier_posterior is None:
            post_all = torch.zeros(B, graph.n_faults, dtype=y.dtype, device=y.device)
            for s in torch.unique(sources):
                sel = sources == s
                post_all[sel] = class_posterior_bma(y[sel], graph, scm, n_graph_samples, source_idx=int(s.item()))
        else:
            post_all = classifier_posterior.to(y.device)
        A_map = graph.final_hard_adjacency()
        W_eff = (A_map * graph.weight_matrix()).detach()
        act = _gate_activations(cfg, y, graph, scm, sources, n_graph_samples, classifier_posterior)
        Ed, Es = causal_disablement_and_sufficiency(y, W_eff, scm, graph.n_faults, activations=act, clamp=clamp)
        scores = wp * post_all + wd * _ed_unit(Ed) + ws * _es_unit(Es)
        conf, pred = scores.max(dim=1)
    return CompositeScores(post_all, Ed, Es, scores, pred, conf)


class TemperatureCalibrator:
    """Multiclass temperature scaling (Guo et al., 2017)."""

    def __init__(self):
        self.temperature: float = 1.0

    @staticmethod
    def _to_logits(probs):
        return torch.log(probs.clamp(min=1e-12))

    def fit(self, probs, labels):
        """probs: (N, K) class-probability vectors.  labels: (N,) int64."""
        logits = self._to_logits(probs.detach())
        logT = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
        nll = nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            loss = nll(logits / torch.exp(logT), labels)
            loss.backward()
            return loss

        try:
            opt.step(closure)
            T = float(torch.exp(logT.detach()).item())
            self.temperature = T if 1e-03 < T < 1e03 else 1.0
        except Exception:
            self.temperature = 1.0

    def transform(self, probs):
        """Return a properly normalised, temperature-scaled probability vector."""
        logits = self._to_logits(probs) / self.temperature
        return torch.softmax(logits, dim=1)


def threshold_sweep(conf, pred, labels, g_lo=0.0, g_hi=1.0, n_grid=401):
    """Return [(gamma, coverage, accuracy_on_accepted, n_errors_accepted), ...]."""
    records = []
    n = int(labels.shape[0])
    for g in torch.linspace(g_lo, g_hi, n_grid):
        accepted = conf >= g
        n_acc = int(accepted.sum().item())
        cov = n_acc / max(n, 1)
        if n_acc == 0:
            records.append((float(g), 0.0, 1.0, 0))
            continue
        corr = pred[accepted] == labels[accepted]
        acc = float(corr.float().mean().item())
        records.append((float(g), cov, acc, int((~corr).sum().item())))
    return records


def optimise_threshold(cfg, conf, pred, labels):
    """Deferral threshold on the calibration split: 'target_cov', 'min_cost' or 'max_cov_acc'."""
    inf = cfg.raw['inference']
    g_lo = float(inf['threshold_grid_min'])
    g_hi = float(inf['threshold_grid_max'])
    n_grid = int(inf['threshold_grid_steps'])
    mode = str(inf.get('threshold_mode', 'target_cov'))
    target_cov = float(inf.get('target_coverage', 0.90))
    records = threshold_sweep(conf, pred, labels, g_lo, g_hi, n_grid)

    if mode == 'target_cov':
        c = conf.detach().cpu().numpy() if hasattr(conf, 'detach') else np.asarray(conf)
        if target_cov >= 1.0:
            return float(c.min())
        return float(np.quantile(c, 1.0 - target_cov, method='lower'))

    if mode == 'min_cost':
        c_err = float(inf.get('cost_error', 10.0))
        c_rev = float(inf.get('cost_review', 1.0))
        n = int(labels.shape[0])
        best_g, best_c = (g_lo, float('inf'))
        for g, cov, acc, n_err in records:
            n_defer = n - int(round(cov * n))
            cost = c_err * n_err + c_rev * n_defer
            if cost < best_c:
                best_c, best_g = (cost, g)
        return best_g

    best_g, best_m = (g_lo, -1.0)
    for g, cov, acc, _ in records:
        if cov * acc > best_m:
            best_m = cov * acc
            best_g = g
    return best_g


def expected_calibration_error(conf, correct, n_bins=10):
    conf_np = conf.detach().cpu().numpy()
    correct_np = correct.detach().cpu().numpy().astype(float)
    total = len(conf_np)
    if total == 0:
        return 0.0
    ece = 0.0
    for b in range(n_bins):
        lo, hi = (b / n_bins, (b + 1) / n_bins)
        in_bin = (conf_np > lo) & (conf_np <= hi)
        if in_bin.sum() == 0:
            continue
        acc = correct_np[in_bin].mean()
        avg = conf_np[in_bin].mean()
        ece += in_bin.sum() / total * abs(acc - avg)
    return float(ece)


@dataclass
class EvalReport:
    accuracy_all: float
    accuracy_accepted: float
    coverage: float
    ece: float
    per_class: Dict[int, Dict[str, float]]
    n_samples: int
    n_accepted: int


def full_evaluation(
    cfg, ds, graph, scm, calibrator, threshold, clf_posterior: Optional[torch.Tensor] = None
) -> EvalReport:
    n_samples = int(cfg.raw['inference']['n_graph_samples'])
    y = ds.gas_values
    labels = ds.labels
    n_bins = int(cfg.raw['inference'].get('ece_bins', 15))
    with torch.no_grad():
        probs = calibrator.transform(clf_posterior) if clf_posterior is not None else None
        cs = composite_scores(cfg, y, graph, scm, n_samples, sources=ds.source, classifier_posterior=probs)
        accept_conf = cs.scores.max(dim=1).values
        accepted = accept_conf >= threshold
        correct = cs.pred == labels
        acc_all = correct.float().mean().item()
        cov = accepted.float().mean().item()
        acc_accepted = correct[accepted].float().mean().item() if accepted.any() else 0.0
        if probs is not None:
            prob_conf, prob_pred = probs.max(dim=1)
            ece = expected_calibration_error(prob_conf, prob_pred == labels, n_bins=n_bins)
        else:
            ece = expected_calibration_error(accept_conf.clamp(0, 1), correct, n_bins=n_bins)
        per_class = {}
        for k in torch.unique(labels).tolist():
            sel = labels == k
            per_class[int(k)] = {
                'accuracy': float(correct[sel].float().mean().item()) if sel.any() else 0.0,
                'coverage': float(accepted[sel].float().mean().item()) if sel.any() else 0.0,
                'accuracy_accepted': (
                    float(correct[sel & accepted].float().mean().item()) if (sel & accepted).any() else 0.0
                ),
                'n': int(sel.sum().item()),
            }
    return EvalReport(
        accuracy_all=acc_all,
        accuracy_accepted=acc_accepted,
        coverage=cov,
        ece=ece,
        per_class=per_class,
        n_samples=int(ds.data.shape[0]),
        n_accepted=int(accepted.sum().item()),
    )
