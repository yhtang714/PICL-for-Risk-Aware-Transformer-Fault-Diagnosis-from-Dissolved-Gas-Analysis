"""Replace the causal model that supplies completion and evidence (NOTEARS, DCDI, class means, unstructured Gaussian) while keeping the read-out fixed.

python experiments/causal_necessity.py --seeds 52 ... 61
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    FAULTS,
    GASES,
    TARGET_COVERAGE,
    accepted_acc_at_coverage,
    aurc,
    composite_from,
    load_seed,
    temperature_apply,
)
from main_table import _dcdi_masked, _forbid_mask, _notears_masked
from picl.augment import impute_training_set
from picl.inference import _ed_unit, _es_unit, causal_disablement_and_sufficiency
from picl.scm import LinearGaussianSCM

N_F, N_G = 6, 5


class FixedGraph:
    """Minimal stand-in for HybridCausalGraph with a fixed weight matrix."""

    def __init__(self, W: torch.Tensor, hard_edges=()):
        self.W = W.clone().float()
        self.n_vars = W.shape[0]
        self.n_faults = N_F
        self.n_gases = N_G
        self.hard_edges = list(hard_edges)
        self.discoverable_edges = []

    def final_hard_adjacency(self):
        return (self.W.abs() > 1e-8).float()

    def weight_matrix(self):
        return self.W


def _scm_from_W(W: np.ndarray, data: np.ndarray) -> LinearGaussianSCM:
    """Linear-Gaussian parameters (mu, diagonal noise) for a fixed W from data."""
    n_vars = W.shape[0]
    scm = LinearGaussianSCM(n_vars=n_vars, n_sources=1, init_log_var=0.0)
    mu = data.mean(0)
    resid = (data - mu) - (data - mu) @ W
    var = resid.var(0) + 1e-4
    with torch.no_grad():
        scm.mu.copy_(torch.from_numpy(mu.astype(np.float32)))
        scm.log_sigma2.copy_(torch.log(torch.from_numpy(var.astype(np.float32))).unsqueeze(0))
    return scm


def _ols_fully_connected(data: np.ndarray) -> np.ndarray:
    """Fully connected DAG: faults -> gases, gas_j -> gas_k for j < k (OLS)."""
    n_vars = data.shape[1]
    W = np.zeros((n_vars, n_vars))
    Xc = data - data.mean(0)
    for k in range(N_F, n_vars):
        parents = list(range(N_F)) + list(range(N_F, k))
        A = Xc[:, parents]
        beta, *_ = np.linalg.lstsq(A, Xc[:, k], rcond=None)
        W[parents, k] = beta
    return W


def _ols_on_support(data: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Least-squares weights on a given fault->gas support (rows = parents)."""
    n_vars = data.shape[1]
    W = np.zeros((n_vars, n_vars))
    Xc = data - data.mean(0)
    for k in range(N_F, n_vars):
        parents = np.where(support[:, k])[0]
        if len(parents) == 0:
            continue
        beta, *_ = np.linalg.lstsq(Xc[:, parents], Xc[:, k], rcond=None)
        W[parents, k] = beta
    return W


def _erh_shd(W: np.ndarray, hard_pairs):
    adj = np.abs(W[:N_F, N_F:]) > 1e-8
    ref = np.zeros_like(adj)
    for f, g in hard_pairs:
        ref[f, g] = True
    erh = float(adj[ref].mean()) if ref.any() else np.nan
    shd = int((adj != ref).sum())
    return erh, shd, int(adj.sum())


def _stage3(b, graph_like, scm_like, source_zero=True):
    """Label-blind completion, head, temperature, composite -- for any linear-Gaussian model."""
    cfg = b.cfg

    def _ds(ds):
        d = ds
        if source_zero:
            from picl.data import PICLDataset

            d = PICLDataset(
                data=ds.data,
                gas_values=ds.gas_values,
                labels=ds.labels,
                source=torch.zeros_like(ds.source),
                miss_mask=ds.miss_mask,
                is_synthetic=ds.is_synthetic,
                log_ppm=ds.log_ppm,
            )
        return impute_training_set(d, graph_like, scm_like, b.log_mu, b.log_sd, blind_faults=True)

    tr, ca, te = _ds(b.train_raw), _ds(b.cal_raw), _ds(b.test_raw)
    from picl.classifier_head import (
        classifier_posterior,
        fit_temperature_and_evidence,
        set_fit_data,
        train_classifier_head,
    )
    from picl.evidence import dataset_log_posterior

    head = train_classifier_head(cfg, tr, graph_like, scm_like)
    reduced = getattr(head, 'missing_mode', 'completed') == 'reduced'
    if reduced:
        set_fit_data(head, tr, ca, N_F)
    with torch.no_grad():
        Pc0 = classifier_posterior(head, ca, graph_like, scm_like).numpy()
    lqc, lqt = dataset_log_posterior(ca, graph_like, scm_like), dataset_log_posterior(te, graph_like, scm_like)
    rows = ca.observed_gases(N_F).all(1).numpy() if reduced else np.ones(len(ca.labels), bool)
    fuse = bool(cfg.raw.get('classifier_head', {}).get('evidence_fusion', False))
    T, gamma = fit_temperature_and_evidence(Pc0[rows], lqc[rows], ca.labels.numpy()[rows], fuse=fuse)
    head.evidence_weight = gamma
    head.temperature_ref = T
    with torch.no_grad():
        Pc = temperature_apply(classifier_posterior(head, ca, graph_like, scm_like).numpy(), T)
        Pt = temperature_apply(classifier_posterior(head, te, graph_like, scm_like).numpy(), T)
    W_eff = (graph_like.final_hard_adjacency() * graph_like.weight_matrix()).detach()
    with torch.no_grad():
        Edc, Esc = causal_disablement_and_sufficiency(ca.gas_values, W_eff, scm_like, N_F)
        Edt, Est = causal_disablement_and_sufficiency(te.gas_values, W_eff, scm_like, N_F)
    Sc = composite_from(Pc, _ed_unit(Edc).numpy(), _es_unit(Esc).numpy())
    St = composite_from(Pt, _ed_unit(Edt).numpy(), _es_unit(Est).numpy())
    return dict(
        Pc=Pc,
        Pt=Pt,
        Sc=Sc,
        St=St,
        yc=ca.labels.numpy(),
        yt=te.labels.numpy(),
        T=T,
        gamma=gamma,
        scm_only_acc=float((lqt.argmax(1) == te.labels.numpy()).mean()),
    )


def _empirical_class_means_model(b):
    """No causal model: class means and pooled covariance as a fully connected
    linear-Gaussian model whose fault->gas block equals the class-mean offsets."""
    tr = b.train
    X = tr.gas_values.numpy().astype(float)
    y = tr.labels.numpy()
    mu_y = X.mean(0)
    n_vars = N_F + N_G
    W = np.zeros((n_vars, n_vars))
    mu_f = np.bincount(y, minlength=N_F) / len(y)
    means = np.array([X[y == k].mean(0) for k in range(N_F)])
    B = means - mu_y[None, :]
    W[:N_F, N_F:] = B
    data = np.column_stack([np.eye(N_F)[y], X])
    scm = _scm_from_W(W, data)
    with torch.no_grad():
        scm.mu.copy_(torch.from_numpy(np.r_[mu_f, mu_y].astype(np.float32)))
    return FixedGraph(torch.from_numpy(W)), scm


def _metrics(res):
    """Metrics of the fused posterior and of the composite score with the intervention channels."""
    corr = res['Pt'].argmax(1) == res['yt']
    corr_comp = res['St'].argmax(1) == res['yt']
    return dict(
        acc_full=float(corr.mean()),
        aurc=aurc(res['Pt'].max(1), corr),
        matched_acc85=accepted_acc_at_coverage(res['Pt'].max(1), corr, TARGET_COVERAGE)['accepted_acc'],
        aurc_with_intervention_channels=aurc(res['St'].max(1), corr_comp),
        scm_only_acc=res['scm_only_acc'],
        temperature=res['T'],
        evidence_weight=res['gamma'],
    )


def main(seeds, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        b = load_seed(seed)
        hard_pairs = [(FAULTS.index(e.src), GASES.index(e.tgt)) for e in b.graph.hard_edges]
        st = torch.load(f'results/seeds/seed_{seed}/models/picl_final_model.pt', map_location='cpu', weights_only=False)
        Xg = st['stage2_real_imputed_gas'].numpy().astype(float)
        yl = st['stage2_real_labels'].numpy()
        data = np.column_stack([np.eye(N_F)[yl], Xg])
        comparators = {}
        comparators['PICL (physics-anchored hybrid graph)'] = (b.graph, b.scm, b.W_eff.numpy())
        W_full = _ols_fully_connected(data)
        comparators['Unstructured Gaussian (fully connected, no physics)'] = (
            FixedGraph(torch.from_numpy(W_full)),
            _scm_from_W(W_full, data),
            W_full,
        )
        mask = _forbid_mask('fault_to_gas_only')
        W_nt = _notears_masked(data.astype(np.float32), mask, seed=seed)
        sup = np.abs(W_nt) > 0.05
        W_nt_ls = _ols_on_support(data, sup)
        comparators['NOTEARS graph (fault->gas)'] = (
            FixedGraph(torch.from_numpy(W_nt_ls)),
            _scm_from_W(W_nt_ls, data),
            W_nt_ls,
        )
        W_dc = _dcdi_masked(data.astype(np.float32), b.train_raw.source.numpy(), mask, seed=seed)
        sup = np.abs(W_dc) > 0.05
        W_dc_ls = _ols_on_support(data, sup)
        comparators['DCDI graph (fault->gas)'] = (
            FixedGraph(torch.from_numpy(W_dc_ls)),
            _scm_from_W(W_dc_ls, data),
            W_dc_ls,
        )
        g_e, s_e = _empirical_class_means_model(b)
        comparators['Empirical class means (no causal model)'] = (g_e, s_e, g_e.W.numpy())
        for name, (g, s, W) in comparators.items():
            res = _stage3(b, g, s, source_zero=(name != 'PICL (physics-anchored hybrid graph)'))
            m = _metrics(res)
            erh, shd, n_edges = _erh_shd(W, hard_pairs)
            rows.append(dict(seed=seed, model=name, ERH=erh, SHD=shd, n_fault_gas_edges=n_edges, **m))
            print(
                f"  s{seed} {name:>55s} acc={m['acc_full']:.4f} aurc={m['aurc']:.5f} acc85={m['matched_acc85']:.4f} scm_only={m['scm_only_acc']:.3f}",
                flush=True,
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'causal_necessity_per_seed.csv', index=False)
    agg = df.groupby('model').agg(['mean', 'std']).drop(columns='seed')
    agg.columns = [f'{a}_{b}' for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out_dir / 'causal_necessity_summary.csv', index=False)
    pd.set_option('display.width', 250)
    print('\n=== Necessity of the causal model (mean over seeds) ===')
    print(
        agg[
            [
                'model',
                'ERH_mean',
                'SHD_mean',
                'n_fault_gas_edges_mean',
                'acc_full_mean',
                'aurc_mean',
                'matched_acc85_mean',
                'aurc_with_intervention_channels_mean',
                'scm_only_acc_mean',
                'evidence_weight_mean',
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f'\nWritten to {out_dir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(52, 62)))
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    main(a.seeds, Path(a.out))
