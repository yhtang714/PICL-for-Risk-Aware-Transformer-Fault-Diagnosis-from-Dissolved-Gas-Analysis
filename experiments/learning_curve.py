"""Retrain on a fraction of each training partition to see how the contribution of the causal model depends on sample size.

python experiments/learning_curve.py --seeds 52 ... 61 --out tables
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    TARGET_COVERAGE,
    _proba6,
    accepted_acc_at_coverage,
    aurc,
    baseline_features,
    baseline_models,
    set_seed,
    temperature_apply,
    temperature_fit,
)
from ablation import variant_full, variant_no_scm_evidence, variant_uniform_prior
from picl.augment import impute_training_set
from picl.classifier_head import classifier_posterior
from picl.data import get_log_stats, load_picl_datasets, subset
from picl.trainer import train_picl

warnings.filterwarnings('ignore')
K = 6
FRACTIONS = [0.25, 0.5, 1.0]
PICL_VARIANTS = [
    ('PICL', variant_full),
    ('PICL, uniform prior', variant_uniform_prior),
    ('PICL without SCM evidence', variant_no_scm_evidence),
]


def stratified_subset(labels, frac, rng):
    keep = np.zeros(len(labels), bool)
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        n = max(3, int(round(frac * len(idx))))
        keep[rng.choice(idx, size=min(n, len(idx)), replace=False)] = True
    return keep


def score(P, y):
    pred = P.argmax(1)
    corr = pred == y
    conf = P.max(1)
    return dict(
        acc=float(corr.mean()),
        macro_f1=float(f1_score(y, pred, average='macro', labels=list(range(K)), zero_division=0)),
        aurc=aurc(conf, corr),
        acc85=accepted_acc_at_coverage(conf, corr, TARGET_COVERAGE)['accepted_acc'],
    )


def main(seeds, out_dir):
    rows = []
    tmp = Path(tempfile.mkdtemp(prefix='lc_'))
    for seed in seeds:
        for frac in FRACTIONS:
            keep = None
            for vname, vfn in PICL_VARIANTS:
                cfg, _ = vfn()
                cfg.raw['experiment']['seed'] = int(seed)
                set_seed(seed)
                train, cal, test = load_picl_datasets(cfg)
                if keep is None:
                    keep = stratified_subset(
                        train.labels.numpy(), frac, np.random.default_rng(1000 * seed + int(100 * frac))
                    )
                tr = subset(train, torch.from_numpy(keep)) if frac < 1.0 else train
                bundle, _ = train_picl(cfg, tr, cal, test, tmp / f's{seed}_{frac}_{vname[:8]}')
                lm, ls = get_log_stats()
                te_i = impute_training_set(test, bundle.graph, bundle.scm, lm, ls, blind_faults=True)
                with torch.no_grad():
                    P = bundle.calibrator.transform(
                        classifier_posterior(bundle.head, te_i, bundle.graph, bundle.scm)
                    ).numpy()
                yt = te_i.labels.numpy()
                rows.append(
                    dict(
                        seed=seed,
                        fraction=frac,
                        n_train=int(keep.sum()) if frac < 1 else len(train.labels),
                        method=vname,
                        **score(P, yt),
                    )
                )
                if vname == 'PICL':
                    tr_i = impute_training_set(tr, bundle.graph, bundle.scm, lm, ls, blind_faults=True)
                    ca_i = impute_training_set(cal, bundle.graph, bundle.scm, lm, ls, blind_faults=True)
                    m = baseline_models(seed)['XGBoost']
                    m.fit(baseline_features(tr_i), tr_i.labels.numpy())
                    pc, pt = _proba6(m, baseline_features(ca_i)), _proba6(m, baseline_features(te_i))
                    T = temperature_fit(pc, ca_i.labels.numpy())
                    rows.append(
                        dict(
                            seed=seed,
                            fraction=frac,
                            n_train=rows[-1]['n_train'],
                            method='XGBoost',
                            **score(temperature_apply(pt, T), yt),
                        )
                    )
            print(
                seed,
                frac,
                {
                    r['method']: (round(r['acc'], 3), round(r['aurc'], 3))
                    for r in rows
                    if r['seed'] == seed and r['fraction'] == frac
                },
                flush=True,
            )
    d = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = 'design' if min(seeds) < 52 else 'eval'
    d.to_csv(out_dir / f'learning_curve_per_seed.csv', index=False)
    agg = d.groupby(['fraction', 'method'])[['n_train', 'acc', 'macro_f1', 'aurc', 'acc85']].agg(['mean', 'std'])
    agg.columns = [f'{a}_{b}' for a, b in agg.columns]
    agg.reset_index().to_csv(out_dir / 'learning_curve_summary.csv', index=False)
    pd.set_option('display.width', 250)
    print(
        d.groupby(['fraction', 'method'])[['n_train', 'acc', 'macro_f1', 'aurc', 'acc85']].mean().round(4).to_string()
    )
    base = d[d.method == 'PICL'].set_index(['seed', 'fraction'])
    for m in d.method.unique():
        if m == 'PICL':
            continue
        o = d[d.method == m].set_index(['seed', 'fraction'])
        diff = (
            (base[['acc', 'aurc', 'acc85']] - o[['acc', 'aurc', 'acc85']])
            .groupby('fraction')
            .agg(['mean', lambda x: int((x > 0).sum())])
        )
        print(f'PICL minus {m}:')
        print(diff.round(4).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 52)))
    ap.add_argument('--out', default='tables/design')
    a = ap.parse_args()
    main(a.seeds, Path(a.out))
