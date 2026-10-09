from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import torch
from .config import PICLConfig
from .data import PICLDataset
from .graph import HybridCausalGraph
from .inference import causal_disablement_and_sufficiency
from .scm import LinearGaussianSCM

_GAS_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]


def _dga_ratios(log_ppm):
    cols = [(log_ppm[:, i] - log_ppm[:, j]).unsqueeze(1) for i, j in _GAS_PAIRS]
    return torch.cat(cols, dim=1)


@dataclass
class ClassifierHead:
    models: list
    n_classes: int
    feature_blocks: Optional[List[str]] = None
    evidence_weight: float = 0.0
    missing_mode: str = 'completed'
    temperature_ref: float = 1.0
    templates: Optional[list] = None
    fit_data: Optional[dict] = None
    pattern_heads: Dict[tuple, tuple] = field(default_factory=dict)
    fuse_evidence: bool = True
    fit_temperature: bool = True


def extract_scm_features(
    ds: PICLDataset,
    graph: HybridCausalGraph,
    scm: LinearGaussianSCM,
    cfg_mode: str = 'onehot',
    feature_blocks: Optional[List[str]] = None,
) -> np.ndarray:
    n_faults = graph.n_faults
    n_vars = graph.n_vars
    y = ds.gas_values
    N = y.shape[0]
    with torch.no_grad():
        A = graph.final_hard_adjacency()
        W_eff = (A * graph.weight_matrix()).detach()
        mu = scm.mu.detach()
        mu_y = mu[n_faults:]
        I_K = torch.eye(n_faults, dtype=W_eff.dtype)
        mus_by_class = scm.intervene_on_faults(W_eff, I_K, n_faults)
        sigma2 = scm.noise_variance(0).detach()
        I = torch.eye(n_vars, dtype=W_eff.dtype)
        T = torch.linalg.solve(I - W_eff, I)
        Sigma = T.T @ torch.diag(sigma2) @ T + 1e-06 * torch.eye(n_vars)
        S_fy = Sigma[:n_faults, n_faults:]
        S_yy = Sigma[n_faults:, n_faults:]
        rhs = (y - mu_y.unsqueeze(0)).T
        Efy = (S_fy @ torch.linalg.solve(S_yy, rhs)).T + mu[:n_faults].unsqueeze(0)
        res = y.unsqueeze(1) - mus_by_class.unsqueeze(0)
        res_norm = torch.norm(res, dim=2)
        post_proxy = Efy.clamp(min=0.0, max=1.0)
        post_proxy = post_proxy / post_proxy.sum(dim=1, keepdim=True).clamp(min=1e-08)
        _mode = str(cfg_mode or 'onehot')
        _act = None if _mode == 'onehot' else post_proxy
        Ed, Es = causal_disablement_and_sufficiency(y, W_eff, scm, n_faults, activations=_act, clamp=False)
        dga = _dga_ratios(ds.log_ppm) if ds.log_ppm is not None else torch.zeros(N, len(_GAS_PAIRS), dtype=y.dtype)
    if feature_blocks is not None and 'loglik' in feature_blocks:
        from .evidence import dataset_log_posterior

        logq = torch.from_numpy(dataset_log_posterior(ds, graph, scm)).to(y.dtype)
    else:
        logq = torch.zeros(N, n_faults, dtype=y.dtype)
    feats = torch.cat([y, Efy, res.reshape(N, -1), res_norm, Ed, Es, dga, logq], dim=1)
    if feature_blocks is not None:
        idx = np.concatenate([np.arange(*FEATURE_BLOCKS[b]) for b in feature_blocks])
        feats = feats[:, torch.from_numpy(idx)]
    else:
        feats = feats[:, :69]
    return feats.cpu().numpy().astype(np.float32)


FEATURE_BLOCKS = {
    'gas': (0, 5),
    'Efy': (5, 11),
    'res': (11, 41),
    'rn': (41, 47),
    'Ed': (47, 53),
    'Es': (53, 59),
    'dga': (59, 69),
    'loglik': (69, 75),
}
ALL_BLOCKS = ['gas', 'Efy', 'res', 'rn', 'Ed', 'Es', 'dga']


def _blocks_from_cfg(cfg) -> Optional[List[str]]:
    hc = cfg.raw.get('classifier_head', {}) if cfg is not None else {}
    fb = hc.get('feature_blocks')
    if fb is None or fb == 'all':
        return None
    return list(fb)


def _build(cfg: PICLConfig) -> List[object]:
    hc = cfg.raw.get('classifier_head', {})
    name = hc.get('model', 'gradient_boosting')
    n_est = int(hc.get('n_estimators', 300))
    max_depth = int(hc.get('max_depth', 4))
    lr = float(hc.get('learning_rate', 0.05))
    seed = int(cfg.raw['experiment']['seed'])
    ensemble = bool(hc.get('ensemble', False))

    cw = hc.get('class_weight')
    n_jobs = int(hc.get('n_jobs', 2))

    def one(name, depth=max_depth):
        if name == 'gradient_boosting':
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(
                max_iter=n_est,
                max_depth=depth if (depth is not None and depth > 0) else None,
                learning_rate=lr,
                random_state=seed,
                class_weight=cw,
            )
        if name == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                n_estimators=n_est,
                max_depth=depth if (depth is not None and depth > 0) else None,
                random_state=seed,
                n_jobs=n_jobs,
                class_weight=cw,
            )
        if name == 'logistic':
            from sklearn.linear_model import LogisticRegression

            return LogisticRegression(max_iter=5000, C=1.0, random_state=seed, class_weight=cw)
        raise ValueError(f'unknown classifier: {name}')

    members = hc.get('members')
    if members:
        return [one(m['model'], int(m.get('max_depth', max_depth))) for m in members]
    if ensemble:
        return [one('gradient_boosting'), one('random_forest'), one('logistic')]
    return [one(name)]


def train_classifier_head(
    cfg: PICLConfig, ds_train: PICLDataset, graph: HybridCausalGraph, scm: LinearGaussianSCM
) -> ClassifierHead:
    hc = cfg.raw.get('classifier_head', {})
    real_only = bool(hc.get('train_on_real_only', False))
    if real_only and ds_train.is_synthetic is not None:
        from .data import subset

        ds_use = subset(ds_train, ~ds_train.is_synthetic)
    else:
        ds_use = ds_train
    mode = str(cfg.raw['inference'].get('intervention_source', 'onehot'))
    blocks = _blocks_from_cfg(cfg)
    missing_mode = str(hc.get('missing_gases', 'completed'))
    if missing_mode == 'reduced' and bool(hc.get('main_on_complete_only', True)):
        from .data import subset

        complete = ds_use.observed_gases(graph.n_faults).all(1)
        ds_use = subset(ds_use, complete)
    X = extract_scm_features(ds_use, graph, scm, cfg_mode=mode, feature_blocks=blocks)
    y = ds_use.labels.cpu().numpy()
    models = _build(cfg)
    from sklearn.base import clone

    templates = [clone(m) for m in models]
    for clf in models:
        clf.fit(X, y)
    return ClassifierHead(
        models=models,
        n_classes=graph.n_faults,
        feature_blocks=blocks,
        missing_mode=missing_mode,
        templates=templates,
        fuse_evidence=bool(hc.get('evidence_fusion', False)),
        fit_temperature=bool(cfg.raw['inference'].get('temperature_scaling', True)),
    )


def classifier_posterior(
    head: ClassifierHead, ds: PICLDataset, graph: HybridCausalGraph, scm: LinearGaussianSCM, cfg_mode: str = 'onehot'
) -> torch.Tensor:
    if getattr(head, 'missing_mode', 'completed') == 'reduced':
        return _posterior_reduced(head, ds, graph, scm, cfg_mode)
    return _posterior_full(head, ds, graph, scm, cfg_mode)


def _posterior_full(
    head: ClassifierHead, ds: PICLDataset, graph: HybridCausalGraph, scm: LinearGaussianSCM, cfg_mode: str = 'onehot'
) -> torch.Tensor:
    X = extract_scm_features(ds, graph, scm, cfg_mode=cfg_mode, feature_blocks=getattr(head, 'feature_blocks', None))
    N = X.shape[0]
    out = np.zeros((N, head.n_classes), dtype=np.float32)
    for clf in head.models:
        _single_thread(clf)
        proba = clf.predict_proba(X)
        for col, c in enumerate(clf.classes_):
            out[:, int(c)] += proba[:, col]
    out /= len(head.models)
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= 0, 1.0, row_sum)
    out = out / row_sum
    gamma = float(getattr(head, 'evidence_weight', 0.0) or 0.0)
    if gamma > 0.0:
        out = fuse_evidence(out, scm_evidence(ds, graph, scm), gamma)
    return torch.from_numpy(out.astype(np.float32))


N_GAS = 5


def pattern_features(gas_z, log_ppm, sources, pat, graph, scm, blocks=('gas', 'dga', 'loglik')):
    """Head inputs and log q restricted to the measured gases in `pat`; other gases are marginalised."""
    blocks = list(blocks) if blocks is not None else ['gas', 'dga', 'loglik']
    from .evidence import scm_log_posterior

    pat = np.asarray(pat, dtype=bool)
    o = np.where(pat)[0]
    g = np.asarray(gas_z, dtype=np.float64)[:, o]
    lp = np.asarray(log_ppm, dtype=np.float64)
    R = [lp[:, i] - lp[:, j] for i, j in _GAS_PAIRS if pat[i] and pat[j]]
    n = g.shape[0]
    obs = torch.from_numpy(np.tile(pat, (n, 1)))
    if pat.any():
        lq = scm_log_posterior(
            torch.as_tensor(np.asarray(gas_z), dtype=torch.float64),
            obs,
            torch.as_tensor(np.asarray(sources)),
            graph,
            scm,
        )
    else:
        lq = np.full((n, graph.n_faults), -np.log(graph.n_faults))
    parts = []
    if 'gas' in blocks:
        parts.append(g)
    if 'dga' in blocks and R:
        parts.append(np.column_stack(R))
    if 'loglik' in blocks:
        parts.append(lq)
    X = np.hstack(parts) if parts and sum(p.shape[1] for p in parts) else np.zeros((n, 1))
    return X.astype(np.float32), lq


def _single_thread(clf):
    if hasattr(clf, 'n_jobs') and clf.n_jobs not in (None, 1):
        clf.n_jobs = 1
    return clf


def _mean_proba(models, X, K):
    out = np.zeros((X.shape[0], K))
    for clf in models:
        _single_thread(clf)
        out[:, np.asarray(clf.classes_).astype(int)] += clf.predict_proba(X)
    out /= len(models)
    s = out.sum(1, keepdims=True)
    return out / np.where(s <= 0, 1.0, s)


def set_fit_data(head: ClassifierHead, train: PICLDataset, cal: PICLDataset, n_faults: int):
    """Keep what is needed to fit the read-out of any pattern of measured gases on demand."""

    def pack(ds):
        return dict(
            gas=ds.gas_values.detach().cpu().numpy().astype(np.float64),
            log_ppm=ds.log_ppm.detach().cpu().numpy().astype(np.float64),
            obs=ds.observed_gases(n_faults).cpu().numpy().astype(bool),
            source=ds.source.cpu().numpy(),
            y=ds.labels.cpu().numpy(),
        )

    head.fit_data = dict(train=pack(train), cal=pack(cal))
    head.pattern_heads = {}


def _fit_pattern(head: ClassifierHead, pat, graph, scm):
    from sklearn.base import clone

    pat = np.asarray(pat, dtype=bool)
    K = head.n_classes
    tr, ca = head.fit_data['train'], head.fit_data['cal']
    rt = tr['obs'][:, pat].all(1)
    rc = ca['obs'][:, pat].all(1)
    blocks = head.feature_blocks
    Xt, _ = pattern_features(tr['gas'][rt], tr['log_ppm'][rt], tr['source'][rt], pat, graph, scm, blocks)
    Xc, lqc = pattern_features(ca['gas'][rc], ca['log_ppm'][rc], ca['source'][rc], pat, graph, scm, blocks)
    models = [clone(m) for m in head.templates]
    for m in models:
        m.fit(Xt, tr['y'][rt])
    Pc = _mean_proba(models, Xc, K)
    T, gamma = fit_temperature_and_evidence(
        Pc,
        lqc,
        ca['y'][rc],
        fuse=bool(getattr(head, 'fuse_evidence', True)),
        fit_temperature=bool(getattr(head, 'fit_temperature', True)),
    )
    return models, T, gamma


def _posterior_reduced(head: ClassifierHead, ds: PICLDataset, graph, scm, cfg_mode='onehot') -> torch.Tensor:
    """Complete records use the main head; other records use the head for their pattern of measured gases."""
    from .data import subset

    K = head.n_classes
    obs = ds.observed_gases(graph.n_faults).cpu().numpy().astype(bool)
    N = obs.shape[0]
    out = np.zeros((N, K), dtype=np.float64)
    full = obs.all(1)
    if full.any():
        sub = subset(ds, torch.from_numpy(full))
        out[full] = _posterior_full(head, sub, graph, scm, cfg_mode).numpy()
    T_ref = float(getattr(head, 'temperature_ref', 1.0) or 1.0)
    gas = ds.gas_values.detach().cpu().numpy().astype(np.float64)
    lp = ds.log_ppm.detach().cpu().numpy().astype(np.float64)
    src = ds.source.cpu().numpy()
    for pat in np.unique(obs[~full], axis=0) if (~full).any() else []:
        rows = (obs == pat).all(1)
        key = tuple(bool(v) for v in pat)
        if key not in head.pattern_heads:
            head.pattern_heads[key] = _fit_pattern(head, pat, graph, scm)
        models, T_p, g_p = head.pattern_heads[key]
        X, lq = pattern_features(gas[rows], lp[rows], src[rows], pat, graph, scm, head.feature_blocks)
        z = (np.log(_mean_proba(models, X, K) + 1e-6) + g_p * lq) / T_p
        z = T_ref * z
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        out[rows] = e / e.sum(1, keepdims=True)
    return torch.from_numpy(out.astype(np.float32))


def scm_evidence(ds, graph, scm) -> np.ndarray:
    from .evidence import dataset_log_posterior

    return dataset_log_posterior(ds, graph, scm)


def fuse_evidence(P: np.ndarray, logq: np.ndarray, gamma: float) -> np.ndarray:
    """softmax(log p_head + gamma * log q_SCM): a product of experts in which the
    SCM's interventional evidence re-weights the head's class probabilities."""
    z = np.log(np.asarray(P, dtype=np.float64) + 1e-6) + gamma * np.asarray(logq, dtype=np.float64)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature_and_evidence(
    P_cal: np.ndarray, logq_cal: np.ndarray, y_cal: np.ndarray, fuse: bool = True, fit_temperature: bool = True
):
    """Fit T and beta >= 0 by NLL on the calibration split. Returns (T, gamma) with gamma = beta * T."""
    from scipy.optimize import minimize
    from scipy.special import log_softmax

    lp = np.log(np.asarray(P_cal, dtype=np.float64) + 1e-6)
    lq = np.asarray(logq_cal, dtype=np.float64)
    y = np.asarray(y_cal)

    def nll(p):
        z = lp / np.exp(p[0]) + (p[1] if fuse else 0.0) * lq
        return -log_softmax(z, axis=1)[np.arange(len(y)), y].mean()

    best = None
    for b0 in ([0.0, 0.3, 1.0] if fuse else [0.0]):
        r = minimize(
            nll,
            x0=[0.0, b0],
            method='L-BFGS-B',
            bounds=[(-3.0, 3.0) if fit_temperature else (0.0, 0.0), (0.0, 3.0) if fuse else (0.0, 0.0)],
        )
        if best is None or r.fun < best.fun:
            best = r
    T = float(np.exp(best.x[0]))
    beta = float(best.x[1]) if fuse else 0.0
    return T, beta * T


def load_or_train_head(cfg, ckpt, ds_train, graph, scm):
    """Use the head stored with the checkpoint; retrain it only if none is stored."""
    head = ckpt.get('classifier_head') if isinstance(ckpt, dict) else None
    if head is not None:
        return head, 'loaded'
    from picl.classifier_head import train_classifier_head

    return train_classifier_head(cfg, ds_train, graph, scm), 'retrained'
