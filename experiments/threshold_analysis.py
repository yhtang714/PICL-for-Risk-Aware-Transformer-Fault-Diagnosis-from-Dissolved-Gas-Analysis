"""Stability of the deferral threshold and expected cost as a function of the cost ratio.

python experiments/threshold_analysis.py --seeds 52 ... 61
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

from picl.augment import impute_training_set
from picl.classifier_head import classifier_posterior, train_classifier_head
from picl.config import load_config
from picl.data import get_log_stats, load_picl_datasets
from picl.graph import HybridCausalGraph
from picl.inference import TemperatureCalibrator, composite_scores, threshold_sweep
from picl.scm import LinearGaussianSCM


def _load_seed(seed):
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
    lm, ls = get_log_stats()
    tr_i = impute_training_set(train, graph, scm, lm, ls, blind_faults=True)
    ca_i = impute_training_set(cal, graph, scm, lm, ls, blind_faults=True)
    te_i = impute_training_set(test, graph, scm, lm, ls, blind_faults=True)
    head = st.get('classifier_head') or train_classifier_head(cfg, tr_i, graph, scm)
    mode = cfg.raw['inference']['intervention_source']
    calib = TemperatureCalibrator()
    p_cal_raw = classifier_posterior(head, ca_i, graph, scm, cfg_mode=mode)
    calib.temperature = float(st['temperature'])
    n_s = int(cfg.raw['inference']['n_graph_samples'])
    cs_cal = composite_scores(
        cfg, ca_i.gas_values, graph, scm, n_s, sources=ca_i.source, classifier_posterior=calib.transform(p_cal_raw)
    )
    p_te = calib.transform(classifier_posterior(head, te_i, graph, scm, cfg_mode=mode))
    cs_te = composite_scores(cfg, te_i.gas_values, graph, scm, n_s, sources=te_i.source, classifier_posterior=p_te)
    return cfg, cs_cal, ca_i.labels, cs_te, te_i.labels


def main(seeds, n_boot, out_dir):
    target_cov = None
    boot_rows, curve_rows, cost_rows = [], [], []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cfg, cs_cal, y_cal, cs_te, y_te = _load_seed(seed)
        target_cov = float(cfg.raw['inference'].get('target_coverage', 0.90))

        conf_c = cs_cal.conf.numpy()
        pred_c = cs_cal.pred.numpy()
        lab_c = y_cal.numpy()
        conf_t = cs_te.conf.numpy()
        pred_t = cs_te.pred.numpy()
        lab_t = y_te.numpy()
        n_cal = len(lab_c)

        rng = np.random.default_rng(seed)
        gammas, covs, accs = [], [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n_cal, n_cal)
            g = float(np.quantile(conf_c[idx], 1.0 - target_cov, method='lower'))
            gammas.append(g)
            acc_mask = conf_t >= g
            covs.append(float(acc_mask.mean()))
            accs.append(float((pred_t[acc_mask] == lab_t[acc_mask]).mean()) if acc_mask.any() else np.nan)
        gammas = np.array(gammas)
        covs = np.array(covs)
        accs = np.array(accs)
        boot_rows.append(
            dict(
                seed=seed,
                n_boot=n_boot,
                target_coverage=target_cov,
                gamma_mean=gammas.mean(),
                gamma_std=gammas.std(),
                gamma_ci_lo=np.percentile(gammas, 2.5),
                gamma_ci_hi=np.percentile(gammas, 97.5),
                test_cov_mean=covs.mean(),
                test_cov_ci_lo=np.percentile(covs, 2.5),
                test_cov_ci_hi=np.percentile(covs, 97.5),
                test_acc_mean=np.nanmean(accs),
                test_acc_ci_lo=np.nanpercentile(accs, 2.5),
                test_acc_ci_hi=np.nanpercentile(accs, 97.5),
            )
        )

        for g, cov, acc, n_err in threshold_sweep(
            torch.from_numpy(conf_t), torch.from_numpy(pred_t), torch.from_numpy(lab_t), n_grid=201
        ):
            curve_rows.append(
                dict(
                    seed=seed,
                    gamma=g,
                    coverage=cov,
                    accepted_accuracy=acc,
                    rejection_rate=1.0 - cov,
                    n_errors_accepted=n_err,
                    n_deferred=int(round((1.0 - cov) * len(lab_t))),
                )
            )

        rec_t = threshold_sweep(torch.from_numpy(conf_t), torch.from_numpy(pred_t), torch.from_numpy(lab_t), n_grid=201)
        n = len(lab_t)
        for ratio in [1, 2, 5, 10, 20, 50, 100]:
            best = None
            for g, cov, acc, n_err in rec_t:
                n_def = n - int(round(cov * n))
                cost = ratio * n_err + 1.0 * n_def
                if best is None or cost < best[0]:
                    best = (cost, g, cov, acc, n_err, n_def)
            cost, g, cov, acc, n_err, n_def = best
            g0, cov0, acc0, err0 = rec_t[0]
            cost_full = ratio * err0
            cost_rows.append(
                dict(
                    seed=seed,
                    cost_ratio=ratio,
                    gamma_opt=g,
                    coverage=cov,
                    accepted_accuracy=acc,
                    n_errors_accepted=n_err,
                    n_deferred=n_def,
                    expected_cost=cost,
                    cost_no_abstention=cost_full,
                    cost_reduction_pct=100.0 * (cost_full - cost) / max(cost_full, 1e-9),
                )
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    bdf = pd.DataFrame(boot_rows)
    bdf.to_csv(out_dir / 'threshold_bootstrap.csv', index=False)
    pd.DataFrame(curve_rows).to_csv(out_dir / 'coverage_accuracy_curve.csv', index=False)
    cdf = pd.DataFrame(cost_rows)
    cdf.to_csv(out_dir / 'expected_cost_curve_test_oracle.csv', index=False)

    print(f'\n=== gamma* bootstrap ({n_boot} calibration resamples per seed, ' f'target coverage {target_cov}) ===')
    print(
        bdf[
            [
                'seed',
                'gamma_mean',
                'gamma_ci_lo',
                'gamma_ci_hi',
                'test_cov_mean',
                'test_acc_mean',
                'test_acc_ci_lo',
                'test_acc_ci_hi',
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(
        f'\npooled gamma* = {bdf.gamma_mean.mean():.4f} '
        f'[{bdf.gamma_ci_lo.mean():.4f}, {bdf.gamma_ci_hi.mean():.4f}]'
    )
    print(
        f'pooled accepted accuracy = {bdf.test_acc_mean.mean():.4f} '
        f'[{bdf.test_acc_ci_lo.mean():.4f}, {bdf.test_acc_ci_hi.mean():.4f}]'
    )

    print('\n=== expected cost, C = r * (accepted errors) + 1 * (deferrals) ===')
    agg = (
        cdf.groupby('cost_ratio')
        .agg(
            gamma=('gamma_opt', 'mean'),
            coverage=('coverage', 'mean'),
            acc=('accepted_accuracy', 'mean'),
            errors=('n_errors_accepted', 'mean'),
            deferred=('n_deferred', 'mean'),
            cost=('expected_cost', 'mean'),
            cost_no_gate=('cost_no_abstention', 'mean'),
            reduction_pct=('cost_reduction_pct', 'mean'),
        )
        .reset_index()
    )
    print(agg.round(3).to_string(index=False))
    print(f'\nWritten to {out_dir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    main(a.seeds, a.n_boot, Path(a.out))
