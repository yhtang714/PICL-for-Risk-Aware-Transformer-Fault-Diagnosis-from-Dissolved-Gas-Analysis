"""Residual diagnostics of the fitted linear-Gaussian SCM.

python experiments/residual_diagnostics.py --seeds 52 ... 61
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from picl.augment import impute_training_set
from picl.config import load_config
from picl.data import get_log_stats, load_picl_datasets
from picl.graph import HybridCausalGraph
from picl.scm import LinearGaussianSCM


def _breusch_pagan(resid, fitted):
    """LM test: regress squared residuals on the fitted value."""
    r2 = resid**2
    x = np.column_stack([np.ones_like(fitted), fitted])
    beta, *_ = np.linalg.lstsq(x, r2, rcond=None)
    pred = x @ beta
    ss_tot = ((r2 - r2.mean()) ** 2).sum()
    ss_res = ((r2 - pred) ** 2).sum()
    if ss_tot <= 0:
        return np.nan, np.nan
    r_squared = 1.0 - ss_res / ss_tot
    lm = len(r2) * r_squared
    return float(lm), float(1.0 - stats.chi2.cdf(lm, df=1))


def main(seeds, out_dir):
    rows, qq_rows = [], []
    for seed in seeds:
        cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
        cfg.raw['experiment']['seed'] = int(seed)
        train, cal, test = load_picl_datasets(cfg)
        graph = HybridCausalGraph(cfg)
        scm = LinearGaussianSCM(
            n_vars=cfg.n_vars,
            n_sources=int(cfg.raw['data']['n_sources']),
            init_log_var=float(cfg.raw['model']['noise_log_var_init']),
        )
        ck = Path(f'results/seeds/seed_{seed}/models/picl_final_model.pt')
        if not ck.exists():
            raise SystemExit(f'Missing {ck}. Run experiments/run_seeds.py first.')
        st = torch.load(ck, map_location='cpu', weights_only=False)
        graph.load_state_dict(st['graph_state'])
        scm.load_state_dict(st['scm_state'])

        lm_, ls_ = get_log_stats()
        tr = impute_training_set(train, graph, scm, lm_, ls_, blind_faults=True)
        with torch.no_grad():
            W = (graph.final_hard_adjacency() * graph.weight_matrix()).detach()
            X = tr.data
            centred = X - scm.mu.unsqueeze(0)
            I = torch.eye(W.shape[0], dtype=W.dtype)
            E = centred @ (I - W)
            fitted = centred @ W
        E = E.numpy()
        fitted = fitted.numpy()

        for j, name in enumerate(cfg.var_names):
            r = E[:, j]
            if np.allclose(r.std(), 0):
                continue
            sw_W, sw_p = stats.shapiro(r[:5000])
            try:
                k2, k2_p = stats.normaltest(r)
            except Exception:
                k2, k2_p = (np.nan, np.nan)
            bp_lm, bp_p = _breusch_pagan(r, fitted[:, j])
            rows.append(
                dict(
                    seed=seed,
                    variable=name,
                    is_gas=name in cfg.gas_names,
                    n=len(r),
                    resid_std=float(r.std()),
                    skew=float(stats.skew(r)),
                    excess_kurtosis=float(stats.kurtosis(r)),
                    shapiro_W=float(sw_W),
                    shapiro_p=float(sw_p),
                    dagostino_K2=float(k2),
                    dagostino_p=float(k2_p),
                    breusch_pagan_LM=bp_lm,
                    breusch_pagan_p=bp_p,
                )
            )
            if seed == seeds[0]:
                z = (r - r.mean()) / (r.std() + 1e-12)
                q_emp = np.sort(z)
                q_theo = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
                step = max(1, len(z) // 400)
                for a, b in zip(q_theo[::step], q_emp[::step]):
                    qq_rows.append(
                        dict(seed=seed, variable=name, theoretical_quantile=float(a), empirical_quantile=float(b))
                    )

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'residual_diagnostics.csv', index=False)
    pd.DataFrame(qq_rows).to_csv(out_dir / 'residual_qq_points.csv', index=False)

    agg = (
        df.groupby(['variable', 'is_gas'])
        .agg(
            skew=('skew', 'mean'),
            exc_kurt=('excess_kurtosis', 'mean'),
            shapiro_p=('shapiro_p', 'mean'),
            dagostino_p=('dagostino_p', 'mean'),
            bp_p=('breusch_pagan_p', 'mean'),
            n_seeds=('seed', 'nunique'),
        )
        .reset_index()
    )
    agg['normality_rejected'] = agg['shapiro_p'] < 0.05
    agg['heteroskedastic'] = agg['bp_p'] < 0.05
    print('\n=== Residual diagnostics (mean over seeds, alpha = 0.05) ===')
    print(agg.round(4).to_string(index=False))

    gas = agg[agg.is_gas]
    print(f'\nGas variables with normality REJECTED : ' f'{int(gas.normality_rejected.sum())} / {len(gas)}')
    print(f'Gas variables with heteroskedasticity : ' f'{int(gas.heteroskedastic.sum())} / {len(gas)}')
    print(
        '\nInterpretation: any rejection means the linear-Gaussian SCM is a working\n'
        'approximation, not a fitted description. Report these tests and frame the\n'
        'assumption as a tractability choice with quantified departure, rather than\n'
        'leaving it unexamined.'
    )
    print(f'\nWritten to {out_dir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    main(a.seeds, Path(a.out))
