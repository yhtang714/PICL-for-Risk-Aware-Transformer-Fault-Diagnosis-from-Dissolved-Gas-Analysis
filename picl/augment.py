from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch
from .config import PICLConfig
from .data import PICLDataset, concat_datasets
from .graph import HybridCausalGraph
from .inference import class_posterior_bma
from .scm import LinearGaussianSCM


def impute_training_set(
    ds,
    graph,
    scm,
    log_mu: Optional[torch.Tensor] = None,
    log_sd: Optional[torch.Tensor] = None,
    blind_faults: bool = False,
) -> PICLDataset:
    """Fill missing gases with the SCM conditional mean; blind_faults=True ignores the fault labels."""
    n_faults = graph.n_faults
    if blind_faults:
        miss_in = ds.miss_mask.clone()
        miss_in[:, :n_faults] = True
        data_in = ds.data.clone()
        data_in[:, :n_faults] = 0.0
    else:
        miss_in = ds.miss_mask
        data_in = ds.data
    with torch.no_grad():
        A = graph.final_hard_adjacency()
        W_eff = (A * graph.weight_matrix()).detach()
        imputed = scm.impute_conditional_mean(data_in, ds.source, miss_in, W_eff)
    if blind_faults:
        imputed[:, :n_faults] = ds.data[:, :n_faults]
    new_log_ppm = None
    if ds.log_ppm is not None:
        new_log_ppm = ds.log_ppm.clone()
        if log_mu is not None and log_sd is not None:
            miss_gas = ds.miss_mask[:, n_faults:]
            if miss_gas.any():
                imputed_gas = imputed[:, n_faults:]
                log_mu_t = log_mu.to(imputed_gas.dtype)
                log_sd_t = log_sd.to(imputed_gas.dtype)
                recovered = imputed_gas * log_sd_t.unsqueeze(0) + log_mu_t.unsqueeze(0)
                new_log_ppm[miss_gas] = recovered[miss_gas]
    return PICLDataset(
        data=imputed,
        gas_values=imputed[:, n_faults:].clone(),
        labels=ds.labels.clone(),
        source=ds.source.clone(),
        miss_mask=torch.zeros_like(ds.miss_mask),
        is_synthetic=ds.is_synthetic.clone(),
        log_ppm=new_log_ppm,
        obs_mask=ds.observed_gases(n_faults).clone(),
    )


def _target_class_counts(labels, target_total, n_classes):
    counts = torch.bincount(labels, minlength=n_classes).float()
    target = (counts / counts.sum() * target_total).round().to(torch.long)
    diff = target_total - int(target.sum().item())
    if diff != 0:
        target[int(target.argmax().item())] += diff
    return target.tolist()


def counterfactual_augment(
    cfg: PICLConfig,
    ds: PICLDataset,
    graph: HybridCausalGraph,
    scm: LinearGaussianSCM,
    target_total: Optional[int] = None,
    rng_seed: int = 0,
) -> Tuple[PICLDataset, List[int]]:
    aug_cfg = cfg.raw['augmentation']
    if target_total is None:
        target_total = int(aug_cfg['target_size'])
    intervention_levels = list(aug_cfg['intervention_levels'])
    r_filter = float(aug_cfg['plausibility_ratio'])
    max_ratio = float(aug_cfg['max_synthetic_ratio'])
    feature_mode = str(cfg.raw['data'].get('gas_feature_mode', 'proportion'))
    enforce_simplex = feature_mode == 'proportion'
    n_faults = graph.n_faults
    n_gases = graph.n_gases
    n_vars = graph.n_vars
    n_bma = int(cfg.raw['inference']['n_graph_samples'])
    rng = torch.Generator().manual_seed(rng_seed)
    with torch.no_grad():
        A_map = graph.final_hard_adjacency()
        W_eff = (A_map * graph.weight_matrix()).detach()
    target_counts = _target_class_counts(ds.labels, target_total, n_faults)
    real_counts = torch.bincount(ds.labels, minlength=n_faults).tolist()
    needed = [max(0, target_counts[k] - real_counts[k]) for k in range(n_faults)]
    cap = [int(max_ratio * real_counts[k]) for k in range(n_faults)]
    needed = [min(needed[k], cap[k]) for k in range(n_faults)]
    synth_rows, synth_labels, synth_sources = ([], [], [])
    accepted_count = [0] * n_faults
    for k in range(n_faults):
        n_need = needed[k]
        if n_need <= 0:
            continue
        class_mask = ds.labels == k
        if class_mask.sum() == 0:
            continue
        source_pool = ds.source[class_mask].cpu().numpy()
        accepted = 0
        attempts = 0
        max_attempts = max(20 * n_need, 500)
        batch_gen = 256
        while accepted < n_need and attempts < max_attempts:
            this_batch = min(batch_gen, (n_need - accepted) * 4)
            attempts += this_batch
            alpha_idx = torch.randint(len(intervention_levels), (this_batch,), generator=rng)
            alpha = torch.tensor([intervention_levels[int(i)] for i in alpha_idx], dtype=ds.data.dtype)
            f_int = torch.zeros(this_batch, n_faults, dtype=ds.data.dtype)
            f_int[:, k] = alpha
            with torch.no_grad():
                mu_gas = scm.intervene_on_faults(W_eff, f_int, n_faults)
                src_sel = np.random.default_rng(rng_seed + k * 1000 + attempts).integers(
                    0, len(source_pool), size=this_batch
                )
                src_idx = torch.from_numpy(source_pool[src_sel].astype(np.int64))
                sigma_y = torch.stack([scm.noise_variance(int(s.item()))[n_faults:] for s in src_idx], dim=0)
                noise = torch.randn(this_batch, n_gases, generator=rng) * torch.sqrt(sigma_y)
                y_cand = mu_gas + noise
                if enforce_simplex:
                    y_cand = y_cand.clamp(min=0.0)
                    row_sum = y_cand.sum(dim=1, keepdim=True)
                    row_sum = torch.where(row_sum <= 0, torch.ones_like(row_sum), row_sum)
                    y_cand = y_cand / row_sum
                post = class_posterior_bma(y_cand, graph, scm, n_samples=n_bma, source_idx=0)
                target_p = post[:, k]
                others = post.clone()
                others[:, k] = -1.0
                keep = target_p >= r_filter * others.max(dim=1).values
                accept_idx = torch.where(keep)[0]
            slots = n_need - accepted
            if len(accept_idx) > slots:
                accept_idx = accept_idx[:slots]
            if len(accept_idx) == 0:
                continue
            y_keep = y_cand[accept_idx]
            onehot = torch.zeros(len(accept_idx), n_faults, dtype=ds.data.dtype)
            onehot[:, k] = 1.0
            synth_rows.append(torch.cat([onehot, y_keep], dim=1))
            synth_labels.append(torch.full((len(accept_idx),), k, dtype=torch.long))
            synth_sources.append(src_idx[accept_idx])
            accepted += len(accept_idx)
            accepted_count[k] = accepted
    if not synth_rows:
        return (ds, accepted_count)
    all_row = torch.cat(synth_rows, dim=0)
    all_lab = torch.cat(synth_labels, dim=0)
    all_src = torch.cat(synth_sources, dim=0)
    synth_gas = all_row[:, n_faults:].clone()
    synth_ds = PICLDataset(
        data=all_row,
        gas_values=synth_gas,
        labels=all_lab,
        source=all_src,
        miss_mask=torch.zeros(all_row.shape[0], n_vars, dtype=torch.bool),
        is_synthetic=torch.ones(all_row.shape[0], dtype=torch.bool),
        log_ppm=synth_gas.clone(),
    )
    ds_marked = PICLDataset(
        data=ds.data,
        gas_values=ds.gas_values,
        labels=ds.labels,
        source=ds.source,
        miss_mask=ds.miss_mask,
        is_synthetic=torch.zeros(ds.data.shape[0], dtype=torch.bool),
        log_ppm=ds.log_ppm,
    )
    return (concat_datasets(ds_marked, synth_ds), accepted_count)


def impute_mixture(ds, graph, scm, log_mu=None, log_sd=None, class_prior=None):
    """Label-blind completion treating the fault block as a discrete latent: the completed value is
    the posterior-weighted conditional mean over the single-fault hypotheses.
    """
    n_faults = graph.n_faults
    K = n_faults
    with torch.no_grad():
        A = graph.final_hard_adjacency()
        W_eff = (A * graph.weight_matrix()).detach()
        n_vars = W_eff.shape[0]
        mus = scm.intervene_on_faults(W_eff, torch.eye(K, dtype=W_eff.dtype), n_faults)
        W_int = W_eff.clone()
        W_int[:, :n_faults] = 0.0
        I = torch.eye(n_vars, dtype=W_eff.dtype)
        T = torch.linalg.solve(I - W_int, I)
        out = ds.data.clone()
        gas = ds.gas_values.clone()
        miss = ds.miss_mask[:, n_faults:]
        prior = torch.full((K,), 1.0 / K, dtype=W_eff.dtype) if class_prior is None else class_prior.to(W_eff.dtype)
        for s in torch.unique(ds.source):
            sigma2 = scm.noise_variance(int(s.item())).detach()
            Sigma_full = T.T @ torch.diag(sigma2) @ T
            Syy = Sigma_full[n_faults:, n_faults:] + 1e-6 * torch.eye(n_vars - n_faults, dtype=W_eff.dtype)
            sel = (ds.source == s).nonzero(as_tuple=True)[0]
            sub_miss = miss[sel]
            uniq, inv = torch.unique(sub_miss.to(torch.int8), dim=0, return_inverse=True)
            for gi in range(uniq.shape[0]):
                m_row = uniq[gi].to(torch.bool)
                if not m_row.any():
                    continue
                o_row = ~m_row
                rows = sel[inv == gi]
                yo = gas[rows][:, o_row]
                Soo = Syy[o_row][:, o_row]
                Smo = Syy[m_row][:, o_row]
                L = torch.linalg.cholesky(Soo)
                logp = []
                for k in range(K):
                    d = yo - mus[k, o_row].unsqueeze(0)
                    z = torch.linalg.solve_triangular(L, d.T, upper=False)
                    quad = (z * z).sum(0)
                    logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
                    logp.append(-0.5 * (quad + logdet) + torch.log(prior[k]))
                logp = torch.stack(logp, dim=1)
                post = torch.softmax(logp, dim=1)
                gain = torch.linalg.solve(Soo, Smo.T).T
                filled = torch.zeros(len(rows), int(m_row.sum()), dtype=W_eff.dtype)
                for k in range(K):
                    cond = mus[k, m_row].unsqueeze(0) + (yo - mus[k, o_row].unsqueeze(0)) @ gain.T
                    filled = filled + post[:, k : k + 1] * cond
                gas[rows.unsqueeze(1), m_row.nonzero(as_tuple=True)[0].unsqueeze(0)] = filled
        out[:, n_faults:] = gas
    new_log_ppm = None
    if ds.log_ppm is not None:
        new_log_ppm = ds.log_ppm.clone()
        if log_mu is not None and log_sd is not None and miss.any():
            recovered = gas * log_sd.to(gas.dtype).unsqueeze(0) + log_mu.to(gas.dtype).unsqueeze(0)
            new_log_ppm[miss] = recovered[miss]
    return PICLDataset(
        data=out,
        gas_values=gas.clone(),
        labels=ds.labels.clone(),
        source=ds.source.clone(),
        miss_mask=torch.zeros_like(ds.miss_mask),
        is_synthetic=ds.is_synthetic.clone(),
        log_ppm=new_log_ppm,
    )
