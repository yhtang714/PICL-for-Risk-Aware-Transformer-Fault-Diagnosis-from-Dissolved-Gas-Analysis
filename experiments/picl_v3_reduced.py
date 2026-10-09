"""Design study: read-out restricted to the measured gases versus completion.

python experiments/picl_v3_reduced.py --seeds 42 ... 51
"""

from __future__ import annotations

import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    GASES,
    TARGET_COVERAGE,
    _proba6,
    accepted_acc_at_coverage,
    aurc,
    baseline_features,
    load_seed,
    picl_channels,
    temperature_apply,
    temperature_fit,
)
from missingness_patterns import _fill, _nan_array, _with_mask
from picl.classifier_head import _GAS_PAIRS
from picl.data import subset
from picl.evidence import scm_log_posterior
from picl_v2_explore import fit_T_beta, fused

warnings.filterwarnings('ignore')
NF, K = 6, 6


def feats(ds, obs_pat, b, with_evidence):
    """Gases in obs_pat, log-ratios among them, and (optionally) log q with the rest marginalised."""
    o = np.where(obs_pat)[0]
    g = ds.gas_values.numpy()[:, o]
    lp = ds.log_ppm.numpy()
    R = [lp[:, i] - lp[:, j] for i, j in _GAS_PAIRS if obs_pat[i] and obs_pat[j]]
    X = [g] + ([np.column_stack(R)] if R else [])
    lq = None
    if with_evidence:
        obs = ds.observed_gases(NF).numpy().astype(bool) & obs_pat[None, :]
        lq = scm_log_posterior(ds.gas_values, torch.from_numpy(obs), ds.source, b.graph, b.scm)
        X.append(lq)
    X = np.hstack(X)
    if X.shape[1] == 0:
        X = np.zeros((len(ds.labels), 1))
    return X, lq


def score(P, y):
    corr = P.argmax(1) == y
    conf = P.max(1)
    return dict(
        acc=float(corr.mean()),
        aurc=aurc(conf, corr),
        acc85=accepted_acc_at_coverage(conf, corr, TARGET_COVERAGE)['accepted_acc'],
    )


def main(seeds, out_dir):
    rows = []
    for seed in seeds:
        b = load_seed(seed)
        rng = np.random.default_rng(seed)
        ytr, yca = b.train.labels.numpy(), b.cal.labels.numpy()
        knn = KNNImputer(n_neighbors=5).fit(_nan_array(b.train_raw, NF))
        rf_full = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=2, class_weight='balanced').fit(
            baseline_features(b.train), ytr
        )
        T_rf = temperature_fit(_proba6(rf_full, baseline_features(b.cal)), yca)
        cache = {}

        def reduced_models(pat):
            key = tuple(pat.tolist())
            if key not in cache:
                Xtr, _ = feats(b.train, pat, b, True)
                Xca, lqc = feats(b.cal, pat, b, True)
                m = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=2, class_weight='balanced').fit(
                    Xtr, ytr
                )
                T, beta = fit_T_beta(np.log(_proba6(m, Xca) + 1e-6), lqc, yca, fuse=True)
                Xtr0, _ = feats(b.train, pat, b, False)
                Xca0, _ = feats(b.cal, pat, b, False)
                m0 = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=2, class_weight='balanced').fit(
                    Xtr0, ytr
                )
                T0 = temperature_fit(_proba6(m0, Xca0), yca)
                cache[key] = (m, T, beta, m0, T0)
            return cache[key]

        te = b.test_raw
        complete = ~te.miss_mask[:, NF:].any(1)
        te_c = subset(b.test, torch.from_numpy(complete.numpy()) if hasattr(complete, 'numpy') else complete)
        te_c_raw = subset(te, complete)
        y = te_c.labels.numpy()
        n = len(y)
        pats = [('none', np.zeros((n, 5), bool))]
        for r in (0.1, 0.3, 0.5):
            for m in range(3):
                pats.append((f'mcar_{int(r*100)}', rng.random((n, 5)) < r))
        for j, gname in enumerate(GASES):
            mk = np.zeros((n, 5), bool)
            mk[:, j] = True
            pats.append((f'single_{gname}', mk))
        for (i, gi), (j, gj) in itertools.combinations(enumerate(GASES), 2):
            mk = np.zeros((n, 5), bool)
            mk[:, [i, j]] = True
            pats.append((f'pair_{gi}_{gj}', mk))
        for name, mk in pats:
            ds_m = _with_mask(te_c_raw, mk) if mk.any() else te_c_raw
            ds_knn = _fill(ds_m, knn.transform(_nan_array(ds_m, NF)), b)
            res = {
                'PICL, kNN completion': picl_channels(b, ds_knn)['P'],
                'RF, kNN completion': temperature_apply(_proba6(rf_full, baseline_features(ds_knn)), T_rf),
            }
            Pp, Pr = np.zeros((n, K)), np.zeros((n, K))
            obs_all = ~mk
            for pat in np.unique(obs_all, axis=0):
                rows_p = (obs_all == pat).all(1)
                m, T, beta, m0, T0 = reduced_models(pat)
                sub = subset(ds_knn, torch.from_numpy(rows_p))
                X, lq = feats(sub, pat, b, True)
                Pp[rows_p] = fused(np.log(_proba6(m, X) + 1e-6), lq, T, beta)
                X0, _ = feats(sub, pat, b, False)
                Pr[rows_p] = temperature_apply(_proba6(m0, X0), T0)
            res['PICL, reduced features (no imputation)'] = Pp
            res['RF, reduced features'] = Pr
            for k_, P in res.items():
                rows.append(dict(seed=seed, pattern=name, method=k_, **score(P, y)))
        print(f'seed {seed} done ({len(cache)} patterns)', flush=True)
    d = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    d.to_csv(out_dir / 'picl_v3_reduced_per_seed.csv', index=False)
    d['family'] = d.pattern.str.split('_').str[0]
    pd.set_option('display.width', 250)
    for metric in ('acc', 'aurc'):
        print(f'\n=== {metric} ===')
        print(d.groupby(['family', 'method'])[metric].mean().unstack().round(4).to_string())
    print(d.groupby(['pattern', 'method']).acc.mean().unstack().round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 52)))
    ap.add_argument('--out', default='tables/design')
    a = ap.parse_args()
    main(a.seeds, Path(a.out))
