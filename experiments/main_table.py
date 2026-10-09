"""Full-coverage comparison of PICL with the probabilistic, causal-DAG, rule-based and discriminative baselines.

python experiments/main_table.py --seeds 52 ... 61
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from picl.config import load_config
from picl.data import load_picl_datasets

FAULT_NAMES = ['PD', 'D1', 'D2', 'T1', 'T2', 'T3']
GAS_NAMES = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
N_F, N_G = 6, 5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _drm_predict(raw):
    """Doernenburg ratio method.  Returns -1 for 'no diagnosis'."""
    out = np.full(len(raw), -1, dtype=int)
    for i, v in enumerate(raw):
        if np.isnan(v).any() or (v <= 0).any():
            continue
        h2, ch4, c2h2, c2h4, c2h6 = v
        r1, r2 = ch4 / h2, c2h2 / c2h4
        r3, r4 = c2h2 / ch4, c2h6 / c2h2
        if r1 > 1.0 and r2 < 0.75 and r3 < 0.3 and r4 > 0.4:
            out[i] = FAULT_NAMES.index('T2')
        elif r1 < 0.1 and r3 < 0.3 and r4 > 0.4:
            out[i] = FAULT_NAMES.index('PD')
        elif 0.1 <= r1 <= 1.0 and r2 >= 0.75 and r3 >= 0.3 and r4 <= 0.4:
            out[i] = FAULT_NAMES.index('D2')
    return out


def _dtm_predict(raw):
    """Canonical Duval Triangle 1 (Duval & DePablo 2001 / IEC 60599 Annex C)."""
    out = np.full(len(raw), -1, dtype=int)
    for i, v in enumerate(raw):
        ch4, c2h4, c2h2 = v[1], v[3], v[2]
        if np.isnan([ch4, c2h4, c2h2]).any():
            continue
        s = ch4 + c2h4 + c2h2
        if s <= 0:
            continue
        p_ch4, p_c2h4, p_c2h2 = 100.0 * ch4 / s, 100.0 * c2h4 / s, 100.0 * c2h2 / s
        if p_ch4 >= 98.0:
            out[i] = FAULT_NAMES.index('PD')
        elif p_c2h2 >= 13.0 and p_c2h4 >= 23.0:
            out[i] = FAULT_NAMES.index('D2')
        elif p_c2h2 >= 13.0:
            out[i] = FAULT_NAMES.index('D1')
        elif p_c2h4 >= 50.0 and p_c2h2 < 15.0:
            out[i] = FAULT_NAMES.index('T3')
        elif p_c2h4 >= 20.0:
            out[i] = FAULT_NAMES.index('T2')
        else:
            out[i] = FAULT_NAMES.index('T1')
    return out


def _forbid_mask(search_space):
    """1 where an edge is ALLOWED, 0 where forbidden.  Order: faults then gases."""
    n = N_F + N_G
    m = np.zeros((n, n), dtype=np.float32)
    m[:N_F, N_F:] = 1.0
    if search_space == 'unrestricted':
        m[N_F:, N_F:] = 1.0
        np.fill_diagonal(m, 0.0)
    return m


def _notears_masked(X, mask, lambda_l1=0.01, rho=1.0, epochs=800, lr=0.02, seed=0):
    """NOTEARS with an explicit allowed-edge mask."""
    torch.manual_seed(seed)
    n, d = X.shape
    Xt = torch.from_numpy(X.astype(np.float32))
    M = torch.from_numpy(mask)
    W = torch.zeros(d, d, requires_grad=True)
    opt = torch.optim.Adam([W], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        Wm = W * M
        resid = Xt - Xt @ Wm
        loss = 0.5 / n * (resid**2).sum() + lambda_l1 * Wm.abs().sum()
        h = torch.trace(torch.matrix_exp(Wm * Wm)) - d
        loss = loss + 0.5 * rho * h * h
        loss.backward()
        opt.step()
    return (W.detach() * M).numpy()


def _dcdi_masked(X, src, mask, **kw):
    """Multi-source NOTEARS variant: shared W, per-source centring."""
    Xc = X.copy()
    for s in np.unique(src):
        sel = src == s
        Xc[sel] = Xc[sel] - Xc[sel].mean(axis=0, keepdims=True)
    return _notears_masked(Xc, mask, **kw)


def _hillclimb_fault_to_gas(X, y, seed=0):
    """Greedy hill-climb restricted to fault->gas (kept for reference)."""
    mask = _forbid_mask('fault_to_gas_only')
    return _notears_masked(X, mask, lambda_l1=0.05, seed=seed)


def _lda_classify_from_W(W, y_gas_tr, lab_tr, y_gas_te):
    """Classify with the class-conditional means implied by the fault->gas block of W."""
    B = np.asarray(W)[:N_F, N_F:]
    resid = y_gas_tr - B[lab_tr]
    var = resid.var(axis=0) + 1e-06
    ll = -0.5 * (((y_gas_te[:, None, :] - B[None, :, :]) ** 2) / var[None, None, :]).sum(axis=2)
    return ll.argmax(axis=1)


def run(seeds, search_space, out_dir, impute='scm'):
    cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
    search_space = search_space or str(cfg.raw.get('baselines', {}).get('dag_search_space', 'fault_to_gas_only'))

    rows = []
    rule_rows = []
    for seed in seeds:
        set_seed(seed)
        cfg.raw['experiment']['seed'] = int(seed)
        train, cal, test = load_picl_datasets(cfg)

        Xtr, ytr = train.gas_values.numpy(), train.labels.numpy()
        Xte, yte = test.gas_values.numpy(), test.labels.numpy()
        Dtr = train.data.numpy()

        if impute == 'scm':
            ck = Path(f'results/seeds/seed_{seed}/models/picl_final_model.pt')
            if ck.exists():
                from picl.data import get_log_stats
                from picl.graph import HybridCausalGraph
                from picl.scm import LinearGaussianSCM

                g = HybridCausalGraph(cfg)
                sc = LinearGaussianSCM(
                    n_vars=cfg.n_vars,
                    n_sources=int(cfg.raw['data']['n_sources']),
                    init_log_var=float(cfg.raw['model']['noise_log_var_init']),
                )
                st = torch.load(ck, map_location='cpu', weights_only=False)
                g.load_state_dict(st['graph_state'])
                sc.load_state_dict(st['scm_state'])
                lm, ls = get_log_stats()
                from _common import baseline_completion

                tr_i = baseline_completion(cfg, train, train, g, sc, lm, ls)
                te_i = baseline_completion(cfg, train, test, g, sc, lm, ls)
                Xtr, Xte = tr_i.gas_values.numpy(), te_i.gas_values.numpy()
                Dtr = tr_i.data.numpy()
            else:
                print(f'  [warn] seed {seed}: no checkpoint, falling back to raw features')

        from sklearn.naive_bayes import GaussianNB
        from sklearn.mixture import GaussianMixture
        from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

        def acc(p):
            return float((p == yte).mean())

        def extra(p):
            return dict(
                bal_acc=float(balanced_accuracy_score(yte, p)),
                macro_f1=float(f1_score(yte, p, average='macro', labels=list(range(N_F)), zero_division=0)),
                mcc=float(matthews_corrcoef(yte, p)),
            )

        def rec(method, family, p, coverage=1.0):
            return dict(seed=seed, method=method, family=family, acc=acc(p), coverage=coverage, **extra(p))

        rows.append(rec('GNB', 'Probabilistic', GaussianNB().fit(Xtr, ytr).predict(Xte)))

        gm_ll = np.zeros((len(Xte), N_F))
        for k in range(N_F):
            sel = ytr == k
            g = GaussianMixture(n_components=2, covariance_type='full', random_state=seed, reg_covar=1e-4).fit(Xtr[sel])
            gm_ll[:, k] = g.score_samples(Xte)
        rows.append(rec('GMM', 'Probabilistic', gm_ll.argmax(axis=1)))

        mask = _forbid_mask(search_space)
        W_nt = _notears_masked(Dtr, mask, seed=seed)
        rows.append(rec(f'NOTEARS [{search_space}]', 'Causal DAG', _lda_classify_from_W(W_nt, Xtr, ytr, Xte)))
        W_dc = _dcdi_masked(Dtr, train.source.numpy(), mask, seed=seed)
        rows.append(rec(f'DCDI* [{search_space}]', 'Causal DAG', _lda_classify_from_W(W_dc, Xtr, ytr, Xte)))
        W_hc = _hillclimb_fault_to_gas(Dtr, ytr, seed=seed)
        rows.append(rec('HillClimb [fault_to_gas_only]', 'Causal DAG', _lda_classify_from_W(W_hc, Xtr, ytr, Xte)))

    for seed in seeds:
        cfg.raw['experiment']['seed'] = int(seed)
        _, _, test_r = load_picl_datasets(cfg)
        raw = np.expm1(test_r.log_ppm.numpy().astype(float))
        raw[test_r.miss_mask[:, N_F:].numpy()] = np.nan
        lab = test_r.labels.numpy()
        for name, fn in (('DRM', _drm_predict), ('DTM', _dtm_predict)):
            p = fn(raw)
            cov_mask = p >= 0
            rule_rows.append(
                dict(
                    seed=seed,
                    method=name,
                    n_total=len(p),
                    n_covered=int(cov_mask.sum()),
                    coverage=float(cov_mask.mean()),
                    acc_on_covered=float((p[cov_mask] == lab[cov_mask]).mean()) if cov_mask.any() else float('nan'),
                    acc_counting_abstention_as_error=float((p == lab).mean()),
                )
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    agg = (
        df.groupby(['method', 'family'])
        .agg(
            acc_mean=('acc', 'mean'),
            acc_std=('acc', 'std'),
            bal_acc_mean=('bal_acc', 'mean'),
            bal_acc_std=('bal_acc', 'std'),
            macro_f1_mean=('macro_f1', 'mean'),
            macro_f1_std=('macro_f1', 'std'),
            mcc_mean=('mcc', 'mean'),
            mcc_std=('mcc', 'std'),
            coverage=('coverage', 'mean'),
            n_seeds=('seed', 'nunique'),
        )
        .reset_index()
        .sort_values('acc_mean', ascending=False)
    )
    agg.to_csv(out_dir / 'main_performance_table.csv', index=False)
    df.to_csv(out_dir / 'main_performance_per_seed.csv', index=False)
    rb = pd.DataFrame(rule_rows)
    rb.to_csv(out_dir / 'rule_based_coverage_breakdown_per_seed.csv', index=False)
    rb = (
        rb.groupby('method')
        .agg(
            n_total=('n_total', 'mean'),
            coverage=('coverage', 'mean'),
            acc_on_covered=('acc_on_covered', 'mean'),
            acc_counting_abstention_as_error=('acc_counting_abstention_as_error', 'mean'),
        )
        .reset_index()
    )
    rb.to_csv(out_dir / 'rule_based_coverage_breakdown.csv', index=False)

    print(f'\nDAG search space: {search_space}   |   imputation: {impute}\n')
    print(agg.to_string(index=False))
    print('\nRule-based methods (abstention reported separately):')
    print(rb.to_string(index=False))
    print(f'\nWritten to {out_dir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(52, 62)))
    ap.add_argument(
        '--dag-search-space',
        default=None,
        choices=['fault_to_gas_only', 'unrestricted'],
        help='edge set searched by the DAG baselines',
    )
    ap.add_argument(
        '--impute', default='scm', choices=['scm', 'none'], help='give baselines the same SCM imputation PICL uses'
    )
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    run(a.seeds, a.dag_search_space, Path(a.out), impute=a.impute)
