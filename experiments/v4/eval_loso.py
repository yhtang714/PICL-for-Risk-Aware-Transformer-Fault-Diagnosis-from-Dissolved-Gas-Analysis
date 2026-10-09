"""Leave-one-source-out evaluation of the unseen-source read-out and the baselines at sub-class and family level.

python experiments/v4/eval_loso.py --seeds 52 53 54 55 56 --out tables
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
from core import aurc, acc_at, fit_T_beta, fused, list_arrays, load, temp_apply, temp_fit, thr_for
from picl.risk import (
    AGG,
    FAMILY,
    SCMArrays,
    ScaleFreeEvidence,
    class_source_weights,
    hierarchical_decide,
    hierarchical_thresholds,
    log_ratios,
    proba_k,
)

K = 6


def unseen_source_readout(d, seed):
    scm = SCMArrays(d['W'], d['mu'], d['log_sigma2'])
    ytr, yca = d['tr_y'], d['ca_y']
    trc, cc = d['tr_obs'].all(1), d['ca_obs'].all(1)
    ev = ScaleFreeEvidence(scm, d['log_sd']).fit(d['tr_gas_c'][trc], d['tr_obs'][trc], d['tr_src'][trc], ytr[trc])
    lq = {s: ev.log_q(d[f'{s}_gas_c'], d[f'{s}_obs'], d[f'{s}_src']) for s in ('tr', 'ca', 'te')}
    X = {s: np.hstack([log_ratios(d[f'{s}_lp_c']), lq[s]]).astype(np.float32) for s in lq}
    m = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=2).fit(
        X['tr'], ytr, sample_weight=class_source_weights(ytr, d['tr_src'])
    )
    Pc, Pt = proba_k(m, X['ca']), proba_k(m, X['te'])
    T, beta = fit_T_beta(Pc[cc], lq['ca'][cc], yca[cc])
    return (
        fused(Pt, lq['te'], T, beta),
        fused(Pc, lq['ca'], T, beta),
        dict(tau=ev.tau.tolist(), tau_s=ev.tau_s, T=T, beta=beta),
    )


def baselines(d, seed):
    f = lambda s: np.hstack([d[f'{s}_gas_c'], log_ratios(d[f'{s}_lp_c'])]).astype(np.float32)
    ytr, yca = d['tr_y'], d['ca_y']
    out = {}
    models = {
        'Random forest': RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=2, class_weight='balanced'),
        'SVM': make_pipeline(
            StandardScaler(), SVC(kernel='rbf', probability=True, random_state=seed, class_weight='balanced')
        ),
        'ANN': make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=seed)
        ),
    }
    for n, m in models.items():
        m.fit(f('tr'), ytr)
        Pc = proba_k(m, f('ca'))
        T = temp_fit(Pc, yca)
        out[n] = (temp_apply(proba_k(m, f('te')), T), temp_apply(Pc, T))
    x = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, random_state=seed, n_jobs=2, eval_metric='mlogloss'
    )
    x.fit(f('tr'), ytr, sample_weight=compute_sample_weight('balanced', ytr))
    Pc = proba_k(x, f('ca'))
    T = temp_fit(Pc, yca)
    out['XGBoost'] = (temp_apply(proba_k(x, f('te')), T), temp_apply(Pc, T))
    return out


def metrics(Pt, Pc, y):
    pred = Pt.argmax(1)
    corr = pred == y
    conf = Pt.max(1)
    t = thr_for(Pc.max(1), 0.85)
    acc_m = conf >= t
    Pf = Pt @ AGG
    fcorr = Pf.argmax(1) == FAMILY[y]
    r = dict(
        acc=corr.mean(),
        macro_f1=f1_score(y, pred, average='macro', labels=list(range(K)), zero_division=0),
        aurc=aurc(conf, corr),
        acc85=acc_at(conf, corr),
        cov_transferred=acc_m.mean(),
        acc_transferred=corr[acc_m].mean() if acc_m.any() else np.nan,
        fam_acc=fcorr.mean(),
        fam_aurc=aurc(Pf.max(1), fcorr),
        fam_acc85=acc_at(Pf.max(1), fcorr),
    )
    t_sub, t_fam = hierarchical_thresholds(Pc, 0.70, 0.95)
    level, p, fp = hierarchical_decide(Pt, t_sub, t_fam)
    sub, fam = level == 2, level == 1
    wrong = (sub & (p != y)) | (fam & (fp != FAMILY[y]))
    r.update(
        h_cov_sub=sub.mean(),
        h_cov_fam=fam.mean(),
        h_cov_total=(sub | fam).mean(),
        h_acc_decided=1 - wrong[sub | fam].mean() if (sub | fam).any() else np.nan,
        h_err_rate=wrong.mean(),
    )
    return r


def main(seeds, out_dir, root='results/v4/loso'):
    rows, params = [], []
    for seed in seeds:
        for fn in list_arrays(root, f'seed{seed}_*'):
            d = load(fn)
            held = str(d['held_out'])
            y = d['te_y']
            res = {'PICL in-distribution read-out': (d['te_P'], d['ca_P'])}
            Pt, Pc, prm = unseen_source_readout(d, seed)
            res['PICL, unseen-source read-out'] = (Pt, Pc)
            res.update(baselines(d, seed))
            for k, (Pt, Pc) in res.items():
                rows.append(dict(seed=seed, held_out=held, n_test=len(y), method=k, **metrics(Pt, Pc, y)))
            params.append(dict(seed=seed, held_out=held, **prm))
            print(
                f'seed {seed} {held:16s} PICL in-distribution {rows[-len(res)]["acc"]:.3f}  unseen-source {rows[-len(res) + 1]["acc"]:.3f}',
                flush=True,
            )
        pd.DataFrame(rows).to_csv(out_dir / 'v4_loso_per_fold.csv', index=False)
        pd.DataFrame(params).to_csv(out_dir / 'v4_loso_params.csv', index=False)
    d = pd.DataFrame(rows)
    M = [c for c in d.columns if c not in ('seed', 'held_out', 'n_test', 'method')]
    micro = (
        d.groupby(['seed', 'method'])
        .apply(lambda x: pd.Series({k: np.average(x[k], weights=x['n_test']) for k in M}), include_groups=False)
        .reset_index()
    )
    micro.to_csv(out_dir / 'v4_loso_micro_per_seed.csv', index=False)
    agg = micro.groupby('method')[M].agg(['mean', 'std'])
    agg.columns = [f'{a}_{b}' for a, b in agg.columns]
    agg.reset_index().to_csv(out_dir / 'v4_loso_micro.csv', index=False)
    pd.set_option('display.width', 250)
    print(micro.groupby('method')[M].mean().round(4).to_string())
    print(d.groupby(['method', 'held_out']).acc.mean().unstack().round(3).to_string())
    print(d.groupby(['method', 'held_out']).fam_acc.mean().unstack().round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[52, 53, 54, 55, 56])
    ap.add_argument('--out', default='tables')
    ap.add_argument('--root', default='results/v4/loso')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    main(a.seeds, out, a.root)
