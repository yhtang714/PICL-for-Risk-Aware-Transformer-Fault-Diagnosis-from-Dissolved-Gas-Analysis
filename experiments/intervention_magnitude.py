"""Model-implied intervention effects compared with the IEC 60599 / Duval reference orderings and with the empirical class means.

python experiments/intervention_magnitude.py --seeds 52 ... 61
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DEFAULT_SEEDS, FAULTS, GASES, load_seed

WITHIN_FAULT_REF = {
    'PD': [('H2', 'CH4'), ('H2', 'C2H2'), ('H2', 'C2H4'), ('H2', 'C2H6'), ('CH4', 'C2H2')],
    'D1': [('C2H2', 'C2H4'), ('H2', 'C2H4'), ('C2H2', 'C2H6'), ('H2', 'C2H6'), ('C2H2', 'CH4')],
    'D2': [('C2H2', 'C2H6'), ('C2H2', 'CH4'), ('H2', 'C2H6'), ('C2H4', 'C2H6'), ('C2H2', 'C2H4')],
    'T1': [('CH4', 'C2H4'), ('CH4', 'C2H2'), ('C2H6', 'C2H2'), ('C2H6', 'C2H4'), ('CH4', 'H2')],
    'T2': [('C2H4', 'C2H2'), ('CH4', 'C2H2'), ('C2H4', 'C2H6'), ('C2H6', 'C2H2'), ('C2H4', 'H2')],
    'T3': [('C2H4', 'C2H2'), ('C2H4', 'CH4'), ('C2H4', 'C2H6'), ('C2H6', 'C2H2'), ('CH4', 'C2H2')],
}
ACROSS_FAULT_REF = {
    'H2': [('PD', 'T1'), ('PD', 'T2'), ('D2', 'T1'), ('D1', 'T1'), ('PD', 'T3')],
    'CH4': [('T1', 'PD'), ('T2', 'PD'), ('T1', 'D2'), ('T2', 'D1'), ('T3', 'D2')],
    'C2H2': [('D2', 'T1'), ('D2', 'T2'), ('D1', 'T1'), ('D2', 'PD'), ('D1', 'PD')],
    'C2H4': [('T3', 'PD'), ('T3', 'T1'), ('T2', 'T1'), ('T3', 'D1'), ('T2', 'PD')],
    'C2H6': [('T1', 'D2'), ('T2', 'D2'), ('T1', 'PD'), ('T3', 'D2'), ('T2', 'PD')],
}


def pair_agreement(values: dict, pairs):
    """Fraction of stated (higher, lower) pairs satisfied and Kendall-style tau."""
    agree = [1.0 if values[h] > values[l] else 0.0 for h, l in pairs]
    return float(np.mean(agree)), float(2 * np.mean(agree) - 1)


def clr(X):
    lg = np.log(np.clip(X, 1e-6, None))
    return lg - lg.mean(axis=1, keepdims=True)


def main(seeds, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    eff_rows, agree_rows, disc_rows = [], [], []
    for seed in seeds:
        b = load_seed(seed)
        nf = 6
        with torch.no_grad():
            mus = b.scm.intervene_on_faults(b.W_eff, torch.eye(6), nf).numpy()
            zero = b.scm.intervene_on_faults(b.W_eff, torch.zeros(1, 6), nf).numpy()[0]
        eff = mus - zero[None, :]
        tr = b.train_raw
        Xz = tr.gas_values.numpy().astype(float).copy()
        Xz[tr.miss_mask[:, nf:].numpy()] = np.nan
        y = tr.labels.numpy()
        emp_z = np.array([np.nanmean(Xz[y == k], 0) for k in range(6)])
        raw = np.expm1(tr.log_ppm.numpy().astype(float))
        raw[tr.miss_mask[:, nf:].numpy()] = np.nan
        comp = ~np.isnan(raw).any(1)
        C = clr(raw[comp])
        emp_clr = np.array([C[y[comp] == k].mean(0) for k in range(6)])
        for k, f in enumerate(FAULTS):
            for j, g in enumerate(GASES):
                eff_rows.append(
                    dict(
                        seed=seed,
                        fault=f,
                        gas=g,
                        model_effect=float(eff[k, j]),
                        model_mean=float(mus[k, j]),
                        empirical_mean_z=float(emp_z[k, j]),
                        empirical_mean_clr=float(emp_clr[k, j]),
                        discrepancy_z=float(mus[k, j] - emp_z[k, j]),
                        hard_edge=any(e.src == f and e.tgt == g for e in b.graph.hard_edges),
                    )
                )
        for k, f in enumerate(FAULTS):
            for space, vals in (
                ('model effect', {g: eff[k, j] for j, g in enumerate(GASES)}),
                ('empirical z-mean', {g: emp_z[k, j] for j, g in enumerate(GASES)}),
                ('empirical CLR-mean', {g: emp_clr[k, j] for j, g in enumerate(GASES)}),
            ):
                frac, tau = pair_agreement(vals, WITHIN_FAULT_REF[f])
                agree_rows.append(
                    dict(
                        seed=seed,
                        scope='within fault',
                        item=f,
                        space=space,
                        agreement=frac,
                        tau=tau,
                        n_pairs=len(WITHIN_FAULT_REF[f]),
                    )
                )
        for j, g in enumerate(GASES):
            for space, vals in (
                ('model effect', {f: eff[k, j] for k, f in enumerate(FAULTS)}),
                ('empirical z-mean', {f: emp_z[k, j] for k, f in enumerate(FAULTS)}),
                ('empirical CLR-mean', {f: emp_clr[k, j] for k, f in enumerate(FAULTS)}),
            ):
                frac, tau = pair_agreement(vals, ACROSS_FAULT_REF[g])
                agree_rows.append(
                    dict(
                        seed=seed,
                        scope='across faults',
                        item=g,
                        space=space,
                        agreement=frac,
                        tau=tau,
                        n_pairs=len(ACROSS_FAULT_REF[g]),
                    )
                )
        disc_rows.append(
            dict(
                seed=seed,
                rms_discrepancy_z=float(np.sqrt(((mus - emp_z) ** 2).mean())),
                rms_discrepancy_hard=float(
                    np.sqrt(
                        np.mean(
                            [
                                (mus[k, j] - emp_z[k, j]) ** 2
                                for k, f in enumerate(FAULTS)
                                for j, g in enumerate(GASES)
                                if any(e.src == f and e.tgt == g for e in b.graph.hard_edges)
                            ]
                        )
                    )
                ),
                rms_discrepancy_nonhard=float(
                    np.sqrt(
                        np.mean(
                            [
                                (mus[k, j] - emp_z[k, j]) ** 2
                                for k, f in enumerate(FAULTS)
                                for j, g in enumerate(GASES)
                                if not any(e.src == f and e.tgt == g for e in b.graph.hard_edges)
                            ]
                        )
                    )
                ),
                spearman_model_vs_empirical=float(stats.spearmanr(mus.ravel(), emp_z.ravel()).correlation),
            )
        )
    de = pd.DataFrame(eff_rows)
    de.to_csv(out_dir / 'intervention_effects_per_seed.csv', index=False)
    de_agg = (
        de.groupby(['fault', 'gas', 'hard_edge'])
        .agg(
            model_effect=('model_effect', 'mean'),
            model_effect_sd=('model_effect', 'std'),
            model_mean=('model_mean', 'mean'),
            empirical_mean_z=('empirical_mean_z', 'mean'),
            empirical_mean_clr=('empirical_mean_clr', 'mean'),
            discrepancy_z=('discrepancy_z', 'mean'),
        )
        .reset_index()
    )
    de_agg.to_csv(out_dir / 'intervention_effects_summary.csv', index=False)
    da = pd.DataFrame(agree_rows)
    da.to_csv(out_dir / 'intervention_ordering_agreement_per_seed.csv', index=False)
    da_agg = (
        da.groupby(['scope', 'item', 'space'])
        .agg(agreement=('agreement', 'mean'), tau=('tau', 'mean'), n_pairs=('n_pairs', 'first'))
        .reset_index()
    )
    da_agg.to_csv(out_dir / 'intervention_ordering_agreement.csv', index=False)
    dd = pd.DataFrame(disc_rows)
    dd.to_csv(out_dir / 'intervention_discrepancy.csv', index=False)
    pd.set_option('display.width', 250)
    print('\n=== Model-implied effect of activating each fault (z units, mean over seeds) ===')
    print(de_agg.pivot(index='fault', columns='gas', values='model_effect').loc[FAULTS, GASES].round(3).to_string())
    print('\n=== Empirical class means, z-scored log level ===')
    print(de_agg.pivot(index='fault', columns='gas', values='empirical_mean_z').loc[FAULTS, GASES].round(3).to_string())
    print('\n=== Empirical class means, CLR composition ===')
    print(
        de_agg.pivot(index='fault', columns='gas', values='empirical_mean_clr').loc[FAULTS, GASES].round(3).to_string()
    )
    print('\n=== Agreement with reference orderings (fraction of stated pairs satisfied) ===')
    print(da_agg.pivot_table(index=['scope', 'item'], columns='space', values='agreement').round(3).to_string())
    print('\nOverall agreement by space:')
    print(da.groupby(['scope', 'space']).agreement.mean().round(3).to_string())
    print('\n=== Discrepancy between model-implied and empirical class means ===')
    print(dd.describe().loc[['mean', 'std']].round(3).to_string())
    print(f'\nWritten to {out_dir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    main(a.seeds, Path(a.out))
