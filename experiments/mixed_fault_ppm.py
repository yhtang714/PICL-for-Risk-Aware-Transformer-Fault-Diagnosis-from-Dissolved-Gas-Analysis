"""Two-fault records built by adding the gas concentrations of two measured records of different classes.

python experiments/mixed_fault_ppm.py --seeds 52 ... 61
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    EVAL_SEEDS,
    TARGET_COVERAGE,
    _proba6,
    fit_calibrated_baselines,
    baseline_features,
    load_seed,
    picl_channels,
    temperature_apply,
    threshold_for_coverage,
)
from picl.data import PICLDataset, subset

LAMS = [0.25, 0.5, 0.75]
N_PAIRS = 40
NF = 6


def mixed_dataset(te, b, ia, ib, lam):
    """ppm-additive mixture of records ia and ib of the complete test set te."""
    ppm_a = torch.expm1(te.log_ppm[ia])
    ppm_b = torch.expm1(te.log_ppm[ib])
    lp = torch.log1p(lam * ppm_a + (1 - lam) * ppm_b)
    g = (lp - b.log_mu.unsqueeze(0)) / b.log_sd.unsqueeze(0)
    n = len(ia)
    data = torch.cat([torch.zeros(n, NF), g], dim=1)
    return PICLDataset(
        data=data,
        gas_values=g.float(),
        labels=te.labels[ia].clone(),
        source=te.source[ia].clone(),
        miss_mask=torch.zeros(n, NF + 5, dtype=torch.bool),
        is_synthetic=torch.zeros(n, dtype=torch.bool),
        log_ppm=lp.float(),
        obs_mask=torch.ones(n, 5, dtype=torch.bool),
    )


def main(seeds, out_dir):
    rows = []
    for seed in seeds:
        b = load_seed(seed)
        rng = np.random.default_rng(seed)
        complete = ~b.test_raw.miss_mask[:, NF:].any(1)
        te = subset(b.test, complete)
        y = te.labels.numpy()
        base = fit_calibrated_baselines(b, names=['Random Forest'], include_gas_only=False)['Random Forest']
        ch_cal = picl_channels(b, b.cal)
        thr_p = threshold_for_coverage(ch_cal['P'].max(1), TARGET_COVERAGE)
        thr_r = threshold_for_coverage(base['P_cal'].max(1), TARGET_COVERAGE)
        P_real = picl_channels(b, te)['P']
        rf_proba = lambda ds: temperature_apply(_proba6(base['model'], baseline_features(ds)), base['T'])
        R_real = rf_proba(te)
        rows.append(
            dict(
                seed=seed,
                method='PICL',
                kind='single fault',
                lam=np.nan,
                accept=float((P_real.max(1) >= thr_p).mean()),
                parent=float((P_real.argmax(1) == y)[P_real.max(1) >= thr_p].mean()),
                third=np.nan,
            )
        )
        rows.append(
            dict(
                seed=seed,
                method='Random forest',
                kind='single fault',
                lam=np.nan,
                accept=float((R_real.max(1) >= thr_r).mean()),
                parent=float((R_real.argmax(1) == y)[R_real.max(1) >= thr_r].mean()),
                third=np.nan,
            )
        )
        for lam in LAMS:
            ia_all, ib_all, pa, pb = [], [], [], []
            for a, c in itertools.combinations(range(6), 2):
                A, B = np.where(y == a)[0], np.where(y == c)[0]
                if len(A) == 0 or len(B) == 0:
                    continue
                ia_all.append(rng.choice(A, N_PAIRS))
                ib_all.append(rng.choice(B, N_PAIRS))
                pa += [a] * N_PAIRS
                pb += [c] * N_PAIRS
            ia, ib = np.concatenate(ia_all), np.concatenate(ib_all)
            pa, pb = np.array(pa), np.array(pb)
            ds = mixed_dataset(te, b, ia, ib, lam)
            for name, P, thr in (('PICL', picl_channels(b, ds)['P'], thr_p), ('Random forest', rf_proba(ds), thr_r)):
                acc = P.max(1) >= thr
                pred = P.argmax(1)
                par = (pred == pa) | (pred == pb)
                rows.append(
                    dict(
                        seed=seed,
                        method=name,
                        kind='two faults',
                        lam=lam,
                        accept=float(acc.mean()),
                        parent=float(par[acc].mean()) if acc.any() else np.nan,
                        third=float((~par)[acc].mean()) if acc.any() else np.nan,
                        misleading_rate=float((acc & ~par).mean()),
                    )
                )
        print(f'seed {seed} done', flush=True)
    d = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_dir / 'mixed_fault_ppm_per_seed.csv', index=False)
    agg = d.groupby(['method', 'kind', 'lam'], dropna=False)[['accept', 'parent', 'third', 'misleading_rate']].agg(
        ['mean', 'std']
    )
    agg.columns = [f'{a}_{b}' for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out_dir / 'mixed_fault_ppm_summary.csv', index=False)
    pd.set_option('display.width', 250)
    print(agg.round(3).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=EVAL_SEEDS)
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    main(a.seeds, Path(a.out))
