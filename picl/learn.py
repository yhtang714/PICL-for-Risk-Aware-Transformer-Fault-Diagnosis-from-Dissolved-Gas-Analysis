from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import torch


def _anneal_tau(epoch, max_epochs, tau0, tau_f):
    progress = min(1.0, max(0.0, epoch / max(1, max_epochs)))
    return tau0 * (tau_f / tau0) ** progress


def _physics_ordering_penalty(cfg, scm, W_eff, n_faults, margin=0.1):
    """Hinge penalty on IEC 60599 gas-dominance orderings of the interventional response."""
    orderings = cfg.raw.get('physics', {}).get('dominance', [])
    if not orderings:
        return torch.zeros((), dtype=W_eff.dtype, device=W_eff.device)
    idx = cfg.var_index
    K = n_faults
    eye = torch.eye(K, dtype=W_eff.dtype, device=W_eff.device)
    mu_gas = scm.intervene_on_faults(W_eff, eye, K)
    pen = torch.zeros((), dtype=W_eff.dtype, device=W_eff.device)
    for o in orderings:
        f = cfg.fault_names.index(o['fault'])
        hi = idx[o['dominant']] - K
        lo = idx[o['weaker']] - K
        pen = pen + torch.relu(mu_gas[f, lo] - mu_gas[f, hi] + margin)
    return pen


def _acyclicity(E_A):
    d = E_A.shape[0]
    return torch.trace(torch.matrix_exp(E_A * E_A)) - d


def _cb_weights(labels, n_classes):
    counts = torch.bincount(labels, minlength=n_classes).float().clamp(min=1.0)
    N = labels.shape[0]
    return N / (float(n_classes) * counts[labels])


def _ce_aux_loss(ds, graph, scm, W_eff, labels, sample_w):
    from .inference import class_posterior_single_graph

    n_faults = graph.n_faults
    mu = scm.mu
    total = torch.zeros((), dtype=W_eff.dtype, device=W_eff.device)
    total_w = torch.zeros((), dtype=W_eff.dtype, device=W_eff.device)
    for s in torch.unique(ds.source):
        sel = ds.source == s
        if sample_w[sel].sum().item() == 0:
            continue
        sigma2 = scm.noise_variance(int(s.item()))
        log_post = class_posterior_single_graph(ds.gas_values[sel], W_eff, sigma2, mu, n_faults)
        nll = -log_post.gather(1, labels[sel].unsqueeze(1)).squeeze(1)
        total = total + (nll * sample_w[sel]).sum()
        total_w = total_w + sample_w[sel].sum()
    if total_w.item() == 0:
        return torch.zeros((), dtype=W_eff.dtype, device=W_eff.device)
    return total / total_w


@dataclass
class LearnResult:
    final_loss: float
    loss_history: List[float]
    acyclicity_history: List[float]
    kept_edges: List[Dict]


def learn_joint(
    cfg, graph, scm, ds, epochs, lr, exclude_synthetic, anneal_tau, phase_name='', use_ce=True, fault_to_gas_pos=False
):
    params = list(graph.parameters()) + list(scm.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    tau0 = float(cfg.raw['model']['temperature_init'])
    tau_f = float(cfg.raw['model']['temperature_final'])
    lam_h = float(cfg.raw['model']['acyclicity_weight'])
    cb = bool(cfg.raw['model'].get('class_balanced_loss', True))
    cb_w = (
        _cb_weights(ds.labels, graph.n_faults).to(ds.data.device)
        if cb
        else torch.ones(ds.data.shape[0], device=ds.data.device)
    )
    sample_w = cb_w.clone()
    if exclude_synthetic:
        sample_w = sample_w * (~ds.is_synthetic).float()
    s_hard = float(cfg.raw['model']['weight_prior_scale_hard'])
    s_plaus = float(cfg.raw['model']['weight_prior_scale_plausible'])
    s_unk = float(cfg.raw['model']['weight_prior_scale_unknown'])
    w_prior_scale = torch.ones(len(graph.discoverable_edges), device=ds.data.device)
    fault_to_gas_mask = torch.zeros(len(graph.discoverable_edges), device=ds.data.device)
    fault_set = set(cfg.fault_names)
    gas_set = set(cfg.gas_names)
    for i, e in enumerate(graph.discoverable_edges):
        w_prior_scale[i] = s_plaus if e.kind == 'plausible' else s_unk
        if e.src in fault_set and e.tgt in gas_set:
            fault_to_gas_mask[i] = 1.0
    n_eff = float(sample_w.sum().item())
    if n_eff == 0:
        raise ValueError(f'{phase_name}: no samples selected')
    beta_kl = float(cfg.raw['model'].get('kl_weight', 1.0))
    w_reg_w = float(cfg.raw['model'].get('weight_reg_weight', 1.0))
    lam_ce = float(cfg.raw['model'].get('discriminative_weight', 0.0)) if use_ce else 0.0
    history, acy_history = ([], [])
    grad_clip = float(cfg.raw['training']['grad_clip'])
    for epoch in range(epochs):
        tau = _anneal_tau(epoch, epochs, tau0, tau_f) if anneal_tau else tau_f
        opt.zero_grad()
        A = graph.sample_adjacency(tau=tau, hard=False)
        W = graph.weight_matrix()
        W_eff = A * W
        ll = scm.log_likelihood(ds.data, ds.source, ds.miss_mask, W_eff, sample_weights=sample_w)
        kl = graph.kl_divergence()
        h = _acyclicity(graph.expected_adjacency())
        if bool(cfg.raw['model'].get('acyclicity_squared', False)):
            h_pen = h * h
        else:
            h_pen = h
        w_reg = 0.5 * (graph.w_disc**2 / w_prior_scale**2).sum()
        noise_reg = float(cfg.raw['model'].get('eta_sigma', 0.01)) * (scm.log_sigma2**2).sum()
        hard_reg = 0.5 * (graph.theta_hard**2).sum() / (s_hard**2 + 1e-08)
        lam_phys = float(cfg.raw['model'].get('physics_ordering_weight', 0.0))
        if lam_phys > 0:
            phys = _physics_ordering_penalty(
                cfg, scm, W_eff, graph.n_faults, float(cfg.raw['model'].get('physics_margin', 0.1))
            )
        else:
            phys = torch.zeros((), dtype=W_eff.dtype, device=W_eff.device)
        if fault_to_gas_pos:
            neg_part = torch.relu(-graph.w_disc) * fault_to_gas_mask
            pos_penalty = 0.5 * neg_part.sum() + 3.0 * (neg_part**2).sum()
        else:
            pos_penalty = torch.zeros((), dtype=W_eff.dtype, device=W_eff.device)
        if lam_ce > 0:
            ce = _ce_aux_loss(ds, graph, scm, W_eff, ds.labels, sample_w)
        else:
            ce = torch.zeros((), dtype=W_eff.dtype, device=W_eff.device)
        loss = (
            -ll / n_eff
            + beta_kl * kl / n_eff
            + lam_h * h_pen
            + lam_phys * phys
            + w_reg_w * (w_reg + noise_reg + hard_reg) / n_eff
            + pos_penalty
            + lam_ce * ce
        )
        if not torch.isfinite(loss):
            continue
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        history.append(float(loss.item()))
        acy_history.append(float(h.item()))
    kept = [r for r in graph.edge_posterior_summary() if r['kept']]
    return LearnResult(
        final_loss=history[-1] if history else float('nan'),
        loss_history=history,
        acyclicity_history=acy_history,
        kept_edges=kept,
    )


def learn_parameters_only(cfg, graph, scm, ds, epochs, lr, phase_name=''):
    graph.a_raw.requires_grad_(False)
    graph.b_raw.requires_grad_(False)
    try:
        trainable = [p for p in graph.parameters() if p.requires_grad] + list(scm.parameters())
        opt = torch.optim.Adam(trainable, lr=lr)
        A_fixed = graph.final_hard_adjacency().detach()
        s_plaus = float(cfg.raw['model']['weight_prior_scale_plausible'])
        s_unk = float(cfg.raw['model']['weight_prior_scale_unknown'])
        s_hard = float(cfg.raw['model']['weight_prior_scale_hard'])
        w_prior_scale = torch.ones(len(graph.discoverable_edges), device=ds.data.device)
        fault_to_gas_mask = torch.zeros(len(graph.discoverable_edges), device=ds.data.device)
        fault_set = set(cfg.fault_names)
        gas_set = set(cfg.gas_names)
        for i, e in enumerate(graph.discoverable_edges):
            w_prior_scale[i] = s_plaus if e.kind == 'plausible' else s_unk
            if e.src in fault_set and e.tgt in gas_set:
                fault_to_gas_mask[i] = 1.0
        grad_clip = float(cfg.raw['training']['grad_clip'])
        n_eff = ds.data.shape[0]
        history = []
        for epoch in range(epochs):
            opt.zero_grad()
            W_eff = A_fixed * graph.weight_matrix()
            ll = scm.log_likelihood(ds.data, ds.source, ds.miss_mask, W_eff)
            noise_reg = float(cfg.raw['model'].get('eta_sigma', 0.01)) * (scm.log_sigma2**2).sum()
            w_reg = 0.5 * (graph.w_disc**2 / w_prior_scale**2).sum()
            hard_reg = 0.5 * (graph.theta_hard**2).sum() / (s_hard**2 + 1e-08)
            neg_part = torch.relu(-graph.w_disc) * fault_to_gas_mask
            pos_penalty = 0.5 * neg_part.sum() + 3.0 * (neg_part**2).sum()
            loss = -(ll / n_eff) + (noise_reg + w_reg + hard_reg) / n_eff + pos_penalty
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            opt.step()
            history.append(float(loss.item()))
        return LearnResult(
            final_loss=history[-1] if history else float('nan'),
            loss_history=history,
            acyclicity_history=[],
            kept_edges=[r for r in graph.edge_posterior_summary() if r['kept']],
        )
    finally:
        graph.a_raw.requires_grad_(True)
        graph.b_raw.requires_grad_(True)
