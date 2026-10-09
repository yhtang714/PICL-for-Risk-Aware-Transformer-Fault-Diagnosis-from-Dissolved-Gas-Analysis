"""Hierarchical decisions and the two-fault screen on the in-distribution splits, for PICL and the calibrated baselines.

python experiments/v4/eval_indist.py --seeds 52 ... 61 --out tables
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings('ignore')
from _common import (
    EVAL_SEEDS,
    _proba6,
    baseline_features,
    fit_calibrated_baselines,
    load_seed,
    picl_channels,
    temperature_apply,
)
from mixed_fault_ppm import LAMS, N_PAIRS, mixed_dataset
from picl.data import subset
from picl.evidence import dataset_log_posterior
from picl.risk import (
    FAMILY,
    PAIRS_CLASS,
    TwoFaultScreen,
    hierarchical_decide,
    hierarchical_thresholds,
    log_ratios,
)

NF = 6
RULES = {'flat 85 %': (0.85, 0.85), 'hierarchical 70/95 %': (0.70, 0.95)}


def head_inputs(b, ds):
    return np.hstack(
        [ds.gas_values.numpy(), log_ratios(ds.log_ppm.numpy()), dataset_log_posterior(ds, b.graph, b.scm)]
    ).astype(np.float32)


def mixture_inputs_fn(b):
    """Read-out inputs of records given by log1p ppm (complete), for the screen's synthetic training mixtures."""
    lm, ls = b.log_mu.numpy().astype(np.float64), b.log_sd.numpy().astype(np.float64)

    def f(lp, src):
        g = ((lp - lm) / ls).astype(np.float32)
        n = len(lp)
        from picl.data import PICLDataset

        ds = PICLDataset(
            data=torch.cat([torch.zeros(n, NF), torch.from_numpy(g)], 1),
            gas_values=torch.from_numpy(g),
            labels=torch.zeros(n, dtype=torch.long),
            source=torch.from_numpy(np.asarray(src, np.int64)),
            miss_mask=torch.zeros(n, NF + 5, dtype=torch.bool),
            is_synthetic=torch.zeros(n, dtype=torch.bool),
            log_ppm=torch.from_numpy(lp.astype(np.float32)),
            obs_mask=torch.ones(n, 5, dtype=torch.bool),
        )
        return head_inputs(b, ds)

    return f


def build_mixtures(b, seed):
    """Identical construction to experiments/mixed_fault_ppm.py."""
    rng = np.random.default_rng(seed)
    complete = ~b.test_raw.miss_mask[:, NF:].any(1)
    te = subset(b.test, complete)
    y = te.labels.numpy()
    out = []
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
        out.append((lam, mixed_dataset(te, b, ia, ib, lam), np.array(pa), np.array(pb)))
    return out


def real_row(level, pred, fpred, y):
    sub, fam = level == 2, level == 1
    dec = sub | fam
    wrong = (sub & (pred != y)) | (fam & (fpred != FAMILY[y]))
    return dict(
        cov_sub=sub.mean(),
        acc_sub=(pred == y)[sub].mean() if sub.any() else np.nan,
        cov_fam=fam.mean(),
        acc_fam=(fpred == FAMILY[y])[fam].mean() if fam.any() else np.nan,
        cov_total=dec.mean(),
        acc_decided=1 - wrong[dec].mean() if dec.any() else np.nan,
        err_rate=wrong.mean(),
        wrong_family=((sub & (FAMILY[pred] != FAMILY[y])) | (fam & (fpred != FAMILY[y]))).mean(),
    )


def mix_row(level, pred, fpred, pa, pb):
    sub, fam = level == 2, level == 1
    par = (pred == pa) | (pred == pb)
    fpar_s = (FAMILY[pred] == FAMILY[pa]) | (FAMILY[pred] == FAMILY[pb])
    fpar_f = (fpred == FAMILY[pa]) | (fpred == FAMILY[pb])
    return dict(
        mix_decided=(sub | fam).mean(),
        mix_sub=sub.mean(),
        mix_fam=fam.mean(),
        mix_misleading=((sub & ~par) | (fam & ~fpar_f)).mean(),
        mix_misleading_sub=(sub & ~par).mean(),
        mix_wrong_family=((sub & ~fpar_s) | (fam & ~fpar_f)).mean(),
    )


def main(seeds, out_dir, lopo=True):
    rows, lopo_rows = [], []
    for seed in seeds:
        b = load_seed(seed)
        yte, yca = b.test.labels.numpy(), b.cal.labels.numpy()
        mixes = build_mixtures(b, seed)
        P = {
            'PICL': (
                picl_channels(b, b.cal)['P'],
                picl_channels(b, b.test)['P'],
                [picl_channels(b, ds)['P'] for _, ds, _, _ in mixes],
            )
        }
        base = fit_calibrated_baselines(b, names=['Random Forest', 'XGBoost', 'SVM'], include_gas_only=False)
        for name, r in base.items():
            P[name] = (
                r['P_cal'],
                r['P_test'],
                [temperature_apply(_proba6(r['model'], baseline_features(ds)), r['T']) for _, ds, _, _ in mixes],
            )
        trc = ~b.train_raw.miss_mask[:, NF:].any(1).numpy()
        fn = mixture_inputs_fn(b)
        X_tr, X_ca, X_te = head_inputs(b, b.train), head_inputs(b, b.cal), head_inputs(b, b.test)
        X_mx = [head_inputs(b, ds) for _, ds, _, _ in mixes]
        lp_tr = b.train_raw.log_ppm.numpy().astype(np.float64)
        scr = (
            TwoFaultScreen(seed)
            .fit(X_tr, lp_tr[trc], b.train_raw.labels.numpy()[trc], b.train_raw.source.numpy()[trc], fn)
            .set_threshold(X_ca)
        )
        flags = (scr.flag(X_ca), scr.flag(X_te), [scr.flag(x) for x in X_mx])
        for method, (Pc, Pt, Pm) in P.items():
            for screen in ((False, True) if method == 'PICL' else (False, True)):
                fc, ft, fm = (flags[0], flags[1], flags[2]) if screen else (None, None, [None] * len(mixes))
                for rule, (c_sub, c_tot) in RULES.items():
                    t_sub, t_fam = hierarchical_thresholds(Pc, c_sub, c_tot, fc)
                    r = dict(seed=seed, method=method, screen=screen, rule=rule, t_sub=t_sub, t_fam=t_fam)
                    r.update(real_row(*hierarchical_decide(Pt, t_sub, t_fam, ft), yte))
                    mrows = [
                        mix_row(*hierarchical_decide(Pm[i], t_sub, t_fam, fm[i]), mixes[i][2], mixes[i][3])
                        for i in range(len(mixes))
                    ]
                    for k in mrows[0]:
                        r[k] = float(np.mean([m[k] for m in mrows]))
                    for i, (lam, _, _, _) in enumerate(mixes):
                        r[f'mix_misleading_{lam}'] = mrows[i]['mix_misleading']
                    rows.append(r)
        if lopo:
            pa_all = np.concatenate([m[2] for m in mixes])
            pb_all = np.concatenate([m[3] for m in mixes])
            Xm_all = np.vstack(X_mx)
            for pair in PAIRS_CLASS:
                sel = (pa_all == pair[0]) & (pb_all == pair[1])
                if not sel.any():
                    continue
                s2 = (
                    TwoFaultScreen(seed)
                    .fit(
                        X_tr,
                        lp_tr[trc],
                        b.train_raw.labels.numpy()[trc],
                        b.train_raw.source.numpy()[trc],
                        fn,
                        exclude_pair=pair,
                    )
                    .set_threshold(X_ca)
                )
                lopo_rows.append(
                    dict(
                        seed=seed,
                        pair=f'{pair[0]}-{pair[1]}',
                        flag_heldout_pair=s2.flag(Xm_all[sel]).mean(),
                        flag_same_pair_full=scr.flag(Xm_all[sel]).mean(),
                        flag_real_test=s2.flag(X_te).mean(),
                    )
                )
        print(f'seed {seed} done', flush=True)
        pd.DataFrame(rows).to_csv(out_dir / 'v4_hierarchical_per_seed.csv', index=False)
        if lopo_rows:
            pd.DataFrame(lopo_rows).to_csv(out_dir / 'v4_screen_lopo_per_seed.csv', index=False)
    d = pd.DataFrame(rows)
    pd.set_option('display.width', 250)
    cols = [
        'cov_sub',
        'acc_sub',
        'cov_fam',
        'acc_fam',
        'cov_total',
        'acc_decided',
        'err_rate',
        'wrong_family',
        'mix_decided',
        'mix_misleading',
        'mix_misleading_sub',
        'mix_wrong_family',
    ]
    print(d.groupby(['method', 'screen', 'rule'])[cols].mean().round(3).to_string())
    if lopo_rows:
        lr = pd.DataFrame(lopo_rows)
        print(
            lr.groupby('pair')[['flag_heldout_pair', 'flag_same_pair_full', 'flag_real_test']]
            .mean()
            .round(3)
            .to_string()
        )
        print(lr[['flag_heldout_pair', 'flag_same_pair_full', 'flag_real_test']].mean().round(3))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=EVAL_SEEDS)
    ap.add_argument('--out', default='tables')
    ap.add_argument('--no-lopo', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    main(a.seeds, out, lopo=not a.no_lopo)
