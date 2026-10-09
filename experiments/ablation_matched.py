"""Component ablation: every variant is retrained and compared with the full model at full and at matched coverage.

python experiments/ablation_matched.py --seeds 52 ... 61
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    COVERAGE_GRID,
    DEFAULT_SEEDS,
    TARGET_COVERAGE,
    accept_stats,
    accepted_acc_at_coverage,
    aurc,
    set_seed,
    threshold_for_coverage,
)
from ablation import VARIANTS
from picl.augment import impute_training_set
from picl.classifier_head import classifier_posterior
from picl.data import get_log_stats, load_picl_datasets
from picl.inference import composite_scores
from picl.trainer import train_picl


def _scores(bundle, cfg, ds):
    """Composite scores of a trained variant on a label-blind completed split."""
    n_s = int(cfg.raw['inference']['n_graph_samples'])
    mode = str(cfg.raw['inference'].get('intervention_source', 'onehot'))
    with torch.no_grad():
        probs = None
        if bundle.head is not None:
            probs = bundle.calibrator.transform(
                classifier_posterior(bundle.head, ds, bundle.graph, bundle.scm, cfg_mode=mode)
            )
        cs = composite_scores(
            cfg, ds.gas_values, bundle.graph, bundle.scm, n_s, sources=ds.source, classifier_posterior=probs
        )
    return cs.conf.numpy(), cs.pred.numpy(), ds.labels.numpy()


def main(seeds, out_root, out_dir, variants=None):
    logging.getLogger().setLevel(logging.WARNING)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, per_sample = [], []
    selected = VARIANTS if not variants else [v for v in VARIANTS if v.__name__ in variants]
    for seed in seeds:
        for vf in selected:
            cfg, label = vf()
            cfg.raw['experiment']['seed'] = int(seed)
            set_seed(int(seed))
            train, cal, test = load_picl_datasets(cfg)
            safe = label.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')
            od = Path(out_root) / f'seed{seed}' / safe
            bundle, summary = train_picl(cfg, train, cal, test, od)
            lm, ls = get_log_stats()
            cal_i = impute_training_set(cal, bundle.graph, bundle.scm, lm, ls, blind_faults=True)
            te_i = impute_training_set(test, bundle.graph, bundle.scm, lm, ls, blind_faults=True)
            cc, pc, yc = _scores(bundle, cfg, cal_i)
            ct, pt, yt = _scores(bundle, cfg, te_i)
            corr_t = pt == yt
            rep = summary['report']['test']
            row = dict(
                seed=seed,
                variant=label,
                acc_all=rep['accuracy_all'],
                acc_accepted_own=rep['accuracy_accepted'],
                coverage_own=rep['coverage'],
                ece=rep['ece'],
                aurc=aurc(ct, corr_t),
            )
            for c in COVERAGE_GRID:
                row[f'matched_acc_{int(c*100)}'] = accepted_acc_at_coverage(ct, corr_t, c)['accepted_acc']
            thr = threshold_for_coverage(cc, TARGET_COVERAGE)
            st = accept_stats(ct, corr_t, thr)
            row['transferred_cov_85'] = st['coverage']
            row['transferred_acc_85'] = st['accepted_acc']
            rows.append(row)
            per_sample.append(pd.DataFrame(dict(seed=seed, variant=label, conf=ct, pred=pt, label=yt)))
            print(
                f"  s{seed} {label:>52s} acc={row['acc_all']:.4f} aurc={row['aurc']:.5f} acc@85={row['matched_acc_85']:.4f}",
                flush=True,
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'ablation_matched_per_seed.csv', index=False)
    pd.concat(per_sample).to_csv(out_dir / 'ablation_matched_scores.csv', index=False)

    full = 'PICL (Full)'
    agg = df.groupby('variant').agg(['mean', 'std']).drop(columns='seed')
    agg.columns = [f'{a}_{b}' for a, b in agg.columns]
    agg = agg.reset_index()
    fm = df[df.variant == full].set_index('seed')
    diffs = []
    for v in df.variant.unique():
        if v == full:
            continue
        d = df[df.variant == v].set_index('seed').loc[fm.index]
        for col in ('acc_all', 'aurc', 'matched_acc_85', 'transferred_acc_85'):
            delta = (d[col] - fm[col]).to_numpy()
            rng = np.random.default_rng(0)
            boots = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(5000)]
            t, p = stats.ttest_rel(d[col], fm[col]) if len(delta) > 1 else (np.nan, np.nan)
            from metrics_full import nb_corrected_t

            _, p_c, ci_c = nb_corrected_t(delta) if len(delta) > 1 else (np.nan, np.nan, (np.nan, np.nan))
            diffs.append(
                dict(
                    variant=v,
                    metric=col,
                    delta=float(delta.mean()),
                    delta_sd=float(delta.std(ddof=1)) if len(delta) > 1 else np.nan,
                    ci_lo=float(np.quantile(boots, 0.025)),
                    ci_hi=float(np.quantile(boots, 0.975)),
                    p_ttest=p,
                    p_corrected=p_c,
                    ci_corrected_lo=ci_c[0],
                    ci_corrected_hi=ci_c[1],
                    sigma_units=float(delta.mean() / (fm[col].std(ddof=1) + 1e-12)),
                )
            )
    dd = pd.DataFrame(diffs)
    dd.to_csv(out_dir / 'ablation_matched_differences.csv', index=False)
    agg.to_csv(out_dir / 'ablation_matched_summary.csv', index=False)
    pd.set_option('display.width', 250)
    show = agg[
        [
            'variant',
            'acc_all_mean',
            'acc_accepted_own_mean',
            'coverage_own_mean',
            'ece_mean',
            'aurc_mean',
            'matched_acc_85_mean',
            'transferred_acc_85_mean',
            'transferred_cov_85_mean',
        ]
    ]
    print('\n=== Ablation at matched coverage (mean over seeds) ===')
    print(show.round(4).to_string(index=False))
    print('\n=== Seed-paired differences from PICL (Full) ===')
    print(dd.round(5).to_string(index=False))
    print(f'\nWritten to {out_dir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    ap.add_argument('--output-root', default='results/ablations')
    ap.add_argument('--out', default='tables')
    ap.add_argument('--variants', nargs='+', default=None)
    a = ap.parse_args()
    main(a.seeds, a.output_root, Path(a.out), a.variants)
