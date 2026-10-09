"""Diagnosis under random deletion, removal of one or two gases, censoring below a detection limit and natural missingness.

python experiments/missingness_patterns.py --seeds 52 ... 61
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    EVAL_SEEDS,
    GASES,
    _BalancedXGB,
    _proba6,
    accept_stats,
    aurc,
    baseline_features,
    load_seed,
    picl_channels,
    temperature_apply,
    temperature_fit,
)
from picl.classifier_head import _GAS_PAIRS
from picl.augment import impute_training_set
from picl.data import PICLDataset

MCAR_RATES = [0.10, 0.20, 0.30, 0.40, 0.50]
CENSOR_PCT = [10, 20, 30]
N_MCAR_MASKS = 5


def _with_mask(ds: PICLDataset, gas_mask: np.ndarray) -> PICLDataset:
    """Return a copy of ds with additional gas entries marked missing (value 0)."""
    nf = ds.data.shape[1] - ds.gas_values.shape[1]
    g = ds.gas_values.clone()
    mm = ds.miss_mask.clone()
    extra = torch.from_numpy(gas_mask) & ~mm[:, nf:]
    mm[:, nf:] = mm[:, nf:] | extra
    g[extra] = 0.0
    d = ds.data.clone()
    d[:, nf:] = g
    lp = ds.log_ppm.clone() if ds.log_ppm is not None else None
    return PICLDataset(
        data=d,
        gas_values=g,
        labels=ds.labels.clone(),
        source=ds.source.clone(),
        miss_mask=mm,
        is_synthetic=ds.is_synthetic.clone(),
        log_ppm=lp,
    )


def _fill(ds: PICLDataset, values: np.ndarray, b) -> PICLDataset:
    """Replace missing gas entries of ds by `values` (z-space) and rebuild log_ppm."""
    nf = b.graph.n_faults
    g = ds.gas_values.clone()
    mm = ds.miss_mask[:, nf:]
    g[mm] = torch.from_numpy(values.astype(np.float32))[mm]
    d = ds.data.clone()
    d[:, nf:] = g
    lp = ds.log_ppm.clone()
    rec = g * b.log_sd.unsqueeze(0) + b.log_mu.unsqueeze(0)
    lp[mm] = rec[mm]
    return PICLDataset(
        data=d,
        gas_values=g,
        labels=ds.labels.clone(),
        source=ds.source.clone(),
        miss_mask=torch.zeros_like(ds.miss_mask),
        is_synthetic=ds.is_synthetic.clone(),
        log_ppm=lp,
        obs_mask=ds.observed_gases(nf).clone(),
    )


def _nan_array(ds: PICLDataset, nf: int) -> np.ndarray:
    X = ds.gas_values.numpy().astype(float).copy()
    X[ds.miss_mask[:, nf:].numpy()] = np.nan
    return X


def _nan_feats(ds: PICLDataset, nf: int) -> np.ndarray:
    """Gases and pairwise log-ratios with NaN wherever a gas was not measured
    (input of XGBoost's native missing-value handling)."""
    from picl.classifier_head import _GAS_PAIRS

    X = _nan_array(ds, nf)
    lp = ds.log_ppm.numpy().astype(float).copy()
    lp[ds.miss_mask[:, nf:].numpy()] = np.nan
    R = np.column_stack([lp[:, i] - lp[:, j] for i, j in _GAS_PAIRS])
    return np.hstack([X, R])


class ReducedForest:
    """Random forest restricted to the measured gases and the log-ratios among them; one model per
    pattern of measured gases, trained and temperature-scaled on the records with those gases."""

    def __init__(self, b, seed):
        self.b, self.seed, self.cache = b, seed, {}

    @staticmethod
    def feats(ds, pat):
        o = np.where(pat)[0]
        lp = ds.log_ppm.numpy()
        R = [lp[:, i] - lp[:, j] for i, j in _GAS_PAIRS if pat[i] and pat[j]]
        X = np.hstack([ds.gas_values.numpy()[:, o]] + ([np.column_stack(R)] if R else []))
        return X if X.shape[1] else np.zeros((len(ds.labels), 1))

    def model(self, pat):
        key = tuple(bool(v) for v in pat)
        if key not in self.cache:
            from picl.data import subset

            b = self.b
            tr_rows = b.train.observed_gases(6)[:, pat].all(1)
            ca_rows = b.cal.observed_gases(6)[:, pat].all(1)
            tr, ca = subset(b.train, tr_rows), subset(b.cal, ca_rows)
            m = RandomForestClassifier(n_estimators=500, random_state=self.seed, n_jobs=2, class_weight='balanced').fit(
                self.feats(tr, pat), tr.labels.numpy()
            )
            m.n_jobs = 1
            T = temperature_fit(_proba6(m, self.feats(ca, pat)), ca.labels.numpy())
            self.cache[key] = (m, T)
        return self.cache[key]

    def predict(self, ds):
        obs = ds.observed_gases(6).numpy().astype(bool)
        P = np.zeros((len(obs), 6))
        for pat in np.unique(obs, axis=0):
            rows = (obs == pat).all(1)
            m, T = self.model(pat)
            from picl.data import subset

            P[rows] = temperature_apply(_proba6(m, self.feats(subset(ds, torch.from_numpy(rows)), pat)), T)
        return P


def main(seeds, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        b = load_seed(seed)
        nf = b.graph.n_faults
        rng = np.random.default_rng(seed)
        Xtr_nan = _nan_array(b.train_raw, nf)
        col_median = np.nanmedian(Xtr_nan, axis=0)
        knn = KNNImputer(n_neighbors=5).fit(Xtr_nan)
        mice = IterativeImputer(random_state=seed, max_iter=20, sample_posterior=False).fit(Xtr_nan)
        rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=2, class_weight='balanced').fit(
            baseline_features(b.train), b.train.labels.numpy()
        )
        rf.n_jobs = 1
        T_rf = temperature_fit(_proba6(rf, baseline_features(b.cal)), b.cal.labels.numpy())
        xgb_native = _BalancedXGB(seed).fit(_nan_feats(b.train_raw, nf), b.train_raw.labels.numpy())
        thr = b.threshold

        te_raw = b.test_raw
        complete = ~te_raw.miss_mask[:, nf:].any(1)
        from picl.data import subset

        te_c = subset(te_raw, complete)
        truth = te_c.gas_values.numpy().copy()
        n, ng = truth.shape
        pct_thr = {j: np.nanpercentile(Xtr_nan[:, j], CENSOR_PCT, axis=0) for j in range(ng)}
        dl_z = {}

        patterns = []
        for r in MCAR_RATES:
            for m in range(N_MCAR_MASKS):
                patterns.append((f'mcar_{int(r*100)}', 'MCAR', r, m, rng.random((n, ng)) < r))
        for j, gname in enumerate(GASES):
            mk = np.zeros((n, ng), bool)
            mk[:, j] = True
            patterns.append((f'single_{gname}', 'single gas', 0.2, 0, mk))
        for (i, gi), (j, gj) in itertools.combinations(enumerate(GASES), 2):
            mk = np.zeros((n, ng), bool)
            mk[:, i] = True
            mk[:, j] = True
            patterns.append((f'pair_{gi}_{gj}', 'two gases', 0.4, 0, mk))
        for pi, p in enumerate(CENSOR_PCT):
            mk = np.zeros((n, ng), bool)
            for j in range(ng):
                mk[:, j] = truth[:, j] < pct_thr[j][pi]
            patterns.append((f'censor_{p}', 'detection-limit censoring', float(mk.mean()), 0, mk))
            dl_z[f'censor_{p}'] = np.array([pct_thr[j][pi] for j in range(ng)])

        rf_red = ReducedForest(b, seed)

        def evaluate(pattern, family, rate, rep, ds_masked, gas_mask, truth_local=None):
            truth_ = truth if truth_local is None else truth_local
            nan_te = _nan_array(ds_masked, nf)
            n_loc = nan_te.shape[0]
            completions = {
                'SCM (label-blind)': impute_training_set(
                    ds_masked, b.graph, b.scm, b.log_mu, b.log_sd, blind_faults=True
                ),
                'column mean': _fill(ds_masked, np.zeros((n_loc, ng)), b),
                'training median': _fill(ds_masked, np.tile(col_median, (n_loc, 1)), b),
                'kNN (k=5)': _fill(ds_masked, knn.transform(nan_te), b),
                'MICE': _fill(ds_masked, mice.transform(nan_te), b),
            }
            y = ds_masked.labels.numpy()

            def picl_row(cname, ds_f, mode):
                old = b.head.missing_mode
                b.head.missing_mode = mode
                try:
                    ch = picl_channels(b, ds_f)
                finally:
                    b.head.missing_mode = old
                corr = ch['pred'] == y
                st = accept_stats(ch['conf'], corr, thr)
                filled = ds_f.gas_values.numpy()
                rmse = (
                    float(np.sqrt(((filled[gas_mask] - truth_[gas_mask]) ** 2).mean()))
                    if (gas_mask.any() and mode == 'completed')
                    else np.nan
                )
                rows.append(
                    dict(
                        seed=seed,
                        pattern=pattern,
                        family=family,
                        rate=rate,
                        rep=rep,
                        completion=cname,
                        downstream='PICL',
                        rmse_log=rmse,
                        acc_full=float(corr.mean()),
                        aurc=aurc(ch['conf'], corr),
                        coverage=st['coverage'],
                        accepted_acc=st['accepted_acc'],
                        n=len(y),
                    )
                )

            picl_row('measured gases only', completions['kNN (k=5)'], 'reduced')
            picl_row('kNN (k=5)', completions['kNN (k=5)'], 'completed')
            picl_row('SCM (label-blind)', completions['SCM (label-blind)'], 'completed')
            for cname, ds_f in completions.items():
                filled = ds_f.gas_values.numpy()
                rmse = float(np.sqrt(((filled[gas_mask] - truth_[gas_mask]) ** 2).mean())) if gas_mask.any() else np.nan
                p_rf = temperature_apply(_proba6(rf, baseline_features(ds_f)), T_rf)
                corr_rf = p_rf.argmax(1) == y
                rows.append(
                    dict(
                        seed=seed,
                        pattern=pattern,
                        family=family,
                        rate=rate,
                        rep=rep,
                        completion=cname,
                        downstream='Random Forest',
                        rmse_log=rmse,
                        acc_full=float(corr_rf.mean()),
                        aurc=aurc(p_rf.max(1), corr_rf),
                        coverage=np.nan,
                        accepted_acc=np.nan,
                        n=len(y),
                    )
                )
            p_red = rf_red.predict(completions['kNN (k=5)'])
            corr_red = p_red.argmax(1) == y
            rows.append(
                dict(
                    seed=seed,
                    pattern=pattern,
                    family=family,
                    rate=rate,
                    rep=rep,
                    completion='measured gases only',
                    downstream='Random Forest',
                    rmse_log=np.nan,
                    acc_full=float(corr_red.mean()),
                    aurc=aurc(p_red.max(1), corr_red),
                    coverage=np.nan,
                    accepted_acc=np.nan,
                    n=len(y),
                )
            )
            if pattern in dl_z:
                dl_lp = dl_z[pattern] * b.log_sd.numpy() + b.log_mu.numpy()
                half = (np.log1p(np.expm1(dl_lp) / 2.0) - b.log_mu.numpy()) / b.log_sd.numpy()
                ds_dl = _fill(ds_masked, np.tile(half, (n_loc, 1)), b)
                ds_dl = PICLDataset(
                    data=ds_dl.data,
                    gas_values=ds_dl.gas_values,
                    labels=ds_dl.labels,
                    source=ds_dl.source,
                    miss_mask=ds_dl.miss_mask,
                    is_synthetic=ds_dl.is_synthetic,
                    log_ppm=ds_dl.log_ppm,
                    obs_mask=torch.ones_like(ds_dl.observed_gases(nf)),
                )
                picl_row('half detection limit', ds_dl, 'reduced')
                p_dl = temperature_apply(_proba6(rf, baseline_features(ds_dl)), T_rf)
                c_dl = p_dl.argmax(1) == y
                rows.append(
                    dict(
                        seed=seed,
                        pattern=pattern,
                        family=family,
                        rate=rate,
                        rep=rep,
                        completion='half detection limit',
                        downstream='Random Forest',
                        rmse_log=np.nan,
                        acc_full=float(c_dl.mean()),
                        aurc=aurc(p_dl.max(1), c_dl),
                        coverage=np.nan,
                        accepted_acc=np.nan,
                        n=len(y),
                    )
                )
            p_x = xgb_native.predict_proba(_nan_feats(ds_masked, nf))
            corr_x = p_x.argmax(1) == y
            rows.append(
                dict(
                    seed=seed,
                    pattern=pattern,
                    family=family,
                    rate=rate,
                    rep=rep,
                    completion='native (NaN)',
                    downstream='XGBoost native missing',
                    rmse_log=np.nan,
                    acc_full=float(corr_x.mean()),
                    aurc=aurc(p_x.max(1), corr_x),
                    coverage=np.nan,
                    accepted_acc=np.nan,
                    n=len(y),
                )
            )

        evaluate('none', 'complete records', 0.0, 0, te_c, np.zeros((n, ng), bool))
        for pattern, family, rate, rep, mk in patterns:
            evaluate(pattern, family, rate, rep, _with_mask(te_c, mk), mk)
        te_m = subset(te_raw, ~complete)
        if len(te_m.labels) > 0:
            evaluate(
                'natural (incomplete records)',
                'natural missingness',
                float(te_m.miss_mask[:, nf:].float().mean()),
                0,
                te_m,
                np.zeros((len(te_m.labels), ng), bool),
                truth_local=te_m.gas_values.numpy(),
            )
        print(f'  seed {seed} done', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'missingness_patterns_per_seed.csv', index=False)
    agg = (
        df.groupby(['family', 'pattern', 'completion', 'downstream'])
        .agg(
            rate=('rate', 'mean'),
            rmse_log=('rmse_log', 'mean'),
            acc_full=('acc_full', 'mean'),
            acc_full_sd=('acc_full', 'std'),
            aurc=('aurc', 'mean'),
            coverage=('coverage', 'mean'),
            accepted_acc=('accepted_acc', 'mean'),
            n=('n', 'mean'),
            n_seeds=('seed', 'nunique'),
        )
        .reset_index()
    )
    agg.to_csv(out_dir / 'missingness_patterns_summary.csv', index=False)
    pd.set_option('display.width', 250)
    sub = agg[agg.downstream == 'PICL']
    print('\n=== PICL head + gate: full-coverage accuracy by completion method ===')
    print(sub.pivot_table(index=['family', 'pattern'], columns='completion', values='acc_full').round(4).to_string())
    print('\n=== PICL (measured gases only): realised coverage under the seed threshold ===')
    print(
        sub[sub.completion == 'measured gases only'][
            ['family', 'pattern', 'rate', 'acc_full', 'aurc', 'coverage', 'accepted_acc']
        ]
        .round(4)
        .to_string(index=False)
    )
    print('\n=== log-space RMSE of removed entries by completion ===')
    print(sub.pivot_table(index=['family', 'pattern'], columns='completion', values='rmse_log').round(3).to_string())
    print('\n=== Random Forest downstream accuracy by completion ===')
    print(
        agg[agg.downstream == 'Random Forest']
        .pivot_table(index=['family', 'pattern'], columns='completion', values='acc_full')
        .round(4)
        .to_string()
    )
    print('\n=== XGBoost native missing handling ===')
    print(
        agg[agg.downstream == 'XGBoost native missing'][['family', 'pattern', 'acc_full']]
        .round(4)
        .to_string(index=False)
    )
    print(f'\nWritten to {out_dir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=EVAL_SEEDS)
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    main(a.seeds, Path(a.out))
