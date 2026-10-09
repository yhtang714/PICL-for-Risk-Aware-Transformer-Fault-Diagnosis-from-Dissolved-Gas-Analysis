"""Design study of the classifier read-out (learner, inputs, use of the SCM evidence).

python experiments/picl_v2_explore.py --seeds 42 ... 51
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import log_softmax, logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    TARGET_COVERAGE,
    accept_stats,
    accepted_acc_at_coverage,
    aurc,
    load_seed,
    picl_channels,
    threshold_for_coverage,
)
from picl.classifier_head import _dga_ratios, extract_scm_features
from picl.data import group_stratified_split
from picl.inference import _ed_unit, _es_unit, causal_disablement_and_sufficiency, expected_calibration_error

warnings.filterwarnings('ignore')
K = 6
LABELS = list(range(K))


def scm_class_loglik(b, raw_ds):
    """l_k(y_o) for every record of a raw (uncompleted) split, missing gases marginalised."""
    W = b.W_eff.double()
    nf = 6
    W_int = W.clone()
    W_int[:, :nf] = 0.0
    I = torch.eye(W.shape[0], dtype=W.dtype)
    T = torch.linalg.solve(I - W_int, I)
    with torch.no_grad():
        means = b.scm.intervene_on_faults(W, torch.eye(nf, dtype=W.dtype), nf).double().numpy()
    y = raw_ds.gas_values.double().numpy()
    miss = raw_ds.miss_mask[:, nf:].numpy()
    src = raw_ds.source.numpy()
    out = np.zeros((len(y), K))
    d_obs = (~miss).sum(1)
    for s in np.unique(src):
        with torch.no_grad():
            sig2 = b.scm.noise_variance(int(s)).double().clone()
        sig2[:nf] = 1e-9
        Sigma = (T.T @ torch.diag(sig2) @ T)[nf:, nf:].numpy()
        for pat in np.unique(miss[src == s], axis=0):
            rows = np.where((src == s) & (miss == pat).all(1))[0]
            o = ~pat
            if not o.any():
                continue
            S = Sigma[np.ix_(o, o)] + 1e-6 * np.eye(o.sum())
            L = np.linalg.cholesky(S)
            logdet = 2 * np.log(np.diag(L)).sum()
            for k in range(K):
                r = y[np.ix_(rows, np.where(o)[0])] - means[k, o]
                z = np.linalg.solve(L, r.T)
                out[rows, k] = -0.5 * ((z * z).sum(0) + logdet + o.sum() * np.log(2 * np.pi))
    return out, d_obs


def head_factories(seed):
    rf = lambda cw: (
        lambda: RandomForestClassifier(
            n_estimators=500, random_state=seed, n_jobs=2, class_weight=cw, min_samples_leaf=1
        )
    )
    return {
        'RF gas+ratios': ('gr', rf(None)),
        'RF gas+ratios (balanced)': ('gr', rf('balanced')),
        'RF gas+ratios+SCM (balanced)': ('grs', rf('balanced')),
        'RF gas+ratios+SCM-loglik (balanced)': ('grl', rf('balanced')),
        'XGB gas+ratios (balanced)': ('gr', 'xgb'),
        'GB(d4) gas+ratios+SCM (balanced) [current]': (
            'grs',
            lambda: HistGradientBoostingClassifier(
                max_iter=300, max_depth=4, learning_rate=0.05, random_state=seed, class_weight='balanced'
            ),
        ),
    }


def fit_predict(factory, Xtr, ytr, Xs, seed):
    if factory == 'xgb':
        m = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, random_state=seed, n_jobs=2, eval_metric='mlogloss'
        )
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight('balanced', ytr))
    else:
        m = factory()
        m.fit(Xtr, ytr)
    outs = []
    for X in Xs:
        P = np.zeros((len(X), K))
        P[:, m.classes_.astype(int)] = m.predict_proba(X)
        outs.append(P)
    return outs


def fit_T_beta(logP, logQ, y, fuse=True):
    def nll(p):
        T, beta = np.exp(p[0]), (p[1] if fuse else 0.0)
        z = logP / T + beta * logQ
        return -(log_softmax(z, axis=1)[np.arange(len(y)), y]).mean()

    best = None
    for b0 in ([0.0, 0.3, 1.0] if fuse else [0.0]):
        r = minimize(nll, x0=[0.0, b0], method='L-BFGS-B', bounds=[(-3, 3), (0.0, 3.0) if fuse else (0.0, 0.0)])
        if best is None or r.fun < best.fun:
            best = r
    return float(np.exp(best.x[0])), float(best.x[1] if fuse else 0.0)


def fused(logP, logQ, T, beta):
    return np.exp(log_softmax(logP / T + beta * logQ, axis=1))


def evaluate(P, score, y, thr):
    pred = P.argmax(1)
    corr = pred == y
    m = accepted_acc_at_coverage(score, corr, TARGET_COVERAGE)
    st = accept_stats(score, corr, thr)
    return dict(
        acc=float(corr.mean()),
        bal_acc=float(balanced_accuracy_score(y, pred)),
        macro_f1=float(f1_score(y, pred, average='macro', labels=LABELS, zero_division=0)),
        aurc=aurc(score, corr),
        acc85_matched=m['accepted_acc'],
        coverage_deployed=st['coverage'],
        acc_deployed=st['accepted_acc'],
        ece=float(expected_calibration_error(torch.from_numpy(P.max(1)), torch.from_numpy(corr), n_bins=15)),
    )


def case_ids(cfg, seed):
    df = pd.read_csv(cfg.raw['data']['csv_path']).query(cfg.raw['data']['filter_query'])
    df = df[df['fault_type'].isin(cfg.raw['data']['fault_types'])].reset_index(drop=True)
    split = group_stratified_split(df['fault_type'].to_numpy(), df['case_id'].to_numpy(), seed)
    return {s: df.loc[split == s, 'case_id'].to_numpy() for s in ('train', 'cal', 'test')}


def main(seeds, out):
    rows, sel_rows = [], []
    for seed in seeds:
        b = load_seed(seed)
        cid = case_ids(b.cfg, seed)
        assert len(cid['train']) == len(b.train.labels) and len(cid['test']) == len(b.test.labels)
        ytr, yca, yte = (d.labels.numpy() for d in (b.train, b.cal, b.test))
        feats = {}
        for name, ds, raw in (('tr', b.train, b.train_raw), ('ca', b.cal, b.cal_raw), ('te', b.test, b.test_raw)):
            gas = ds.gas_values.numpy()
            rat = _dga_ratios(ds.log_ppm).numpy()
            scm = extract_scm_features(ds, b.graph, b.scm, feature_blocks=['Efy', 'rn', 'Ed', 'Es'])
            ll, dobs = scm_class_loglik(b, raw)
            logq = ll - logsumexp(ll, axis=1, keepdims=True)
            with torch.no_grad():
                Ed, Es = causal_disablement_and_sufficiency(ds.gas_values, b.W_eff, b.scm, 6)
            feats[name] = dict(
                gr=np.hstack([gas, rat]),
                grs=np.hstack([gas, rat, scm]),
                grl=np.hstack([gas, rat, logq]),
                logq=logq,
                typ=ll.max(1) / np.maximum(dobs, 1),
                Edu=_ed_unit(Ed).numpy()[:, 0],
                Esu=_es_unit(Es).numpy(),
            )
        ch_c, ch_t = picl_channels(b, b.cal), picl_channels(b, b.test)
        thr = threshold_for_coverage(ch_c['conf'], TARGET_COVERAGE)
        rows.append(
            dict(seed=seed, system='PICL (as configured)', **evaluate(np.eye(K)[ch_t['pred']], ch_t['conf'], yte, thr))
        )
        rows[-1]['ece'] = float(
            expected_calibration_error(
                torch.from_numpy(ch_t['P'].max(1)), torch.from_numpy(ch_t['P'].argmax(1) == yte), n_bins=15
            )
        )
        heads = head_factories(seed)
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = {h: np.zeros((len(ytr), K)) for h in heads}
        for tr_i, va_i in cv.split(feats['tr']['gr'], ytr, cid['train']):
            for h, (fs, fac) in heads.items():
                oof[h][va_i] = fit_predict(fac, feats['tr'][fs][tr_i], ytr[tr_i], [feats['tr'][fs][va_i]], seed)[0]
        cv_f1 = {h: f1_score(ytr, oof[h].argmax(1), average='macro', labels=LABELS, zero_division=0) for h in heads}
        best_f1 = max(cv_f1.values())
        chosen = next(h for h in heads if cv_f1[h] >= best_f1 - 0.005)
        per_head = {}
        for h, (fs, fac) in heads.items():
            Pc, Pt = fit_predict(fac, feats['tr'][fs], ytr, [feats['ca'][fs], feats['te'][fs]], seed)
            lPc, lPt = np.log(Pc + 1e-6), np.log(Pt + 1e-6)
            T0, _ = fit_T_beta(lPc, feats['ca']['logq'], yca, fuse=False)
            T1, beta = fit_T_beta(lPc, feats['ca']['logq'], yca, fuse=True)
            lPo = np.log(oof[h] + 1e-6)
            per_head[h] = dict(
                T0=T0,
                T1=T1,
                beta=beta,
                P0=(fused(lPc, 0, T0, 0), fused(lPt, 0, T0, 0)),
                P1=(fused(lPc, feats['ca']['logq'], T1, beta), fused(lPt, feats['te']['logq'], T1, beta)),
                Po1=fused(lPo, feats['tr']['logq'], T1, beta),
            )
            for tag, (Pcc, Ptt) in (('head', per_head[h]['P0']), ('head+SCM evidence', per_head[h]['P1'])):
                thr_h = threshold_for_coverage(Pcc.max(1), TARGET_COVERAGE)
                rows.append(
                    dict(seed=seed, system=f'{h} | {tag} | confidence', **evaluate(Ptt, Ptt.max(1), yte, thr_h))
                )
        ph = per_head[chosen]
        Pc, Pt, Po = ph['P1'][0], ph['P1'][1], ph['Po1']

        def gate_feats(P, f):
            s = np.sort(P, 1)
            pred = P.argmax(1)
            return np.column_stack(
                [s[:, -1], s[:, -1] - s[:, -2], f['typ'], f['Edu'], f['Esu'][np.arange(len(P)), pred]]
            )

        rej = LogisticRegression(max_iter=2000, C=1.0).fit(gate_feats(Po, feats['tr']), Po.argmax(1) == ytr)

        def scores(P, f):
            pred = P.argmax(1)
            comp = 0.4 * P[np.arange(len(P)), pred] + 0.3 * f['Edu'] + 0.3 * f['Esu'][np.arange(len(P)), pred]
            return {
                'confidence': P.max(1),
                'composite (fixed weights)': comp,
                'learned deferral': rej.predict_proba(gate_feats(P, f))[:, 1],
            }

        Sc, St = scores(Pc, feats['ca']), scores(Pt, feats['te'])
        cal_aurc = {g: aurc(Sc[g], Pc.argmax(1) == yca) for g in Sc}
        gate = min(cal_aurc, key=cal_aurc.get)
        for g in Sc:
            thr_g = threshold_for_coverage(Sc[g], TARGET_COVERAGE)
            rows.append(
                dict(seed=seed, system=f'nested head + SCM evidence | gate: {g}', **evaluate(Pt, St[g], yte, thr_g))
            )
        thr_g = threshold_for_coverage(Sc[gate], TARGET_COVERAGE)
        rows.append(
            dict(seed=seed, system='PICL-v2 (nested: head, SCM evidence, gate)', **evaluate(Pt, St[gate], yte, thr_g))
        )
        P0c, P0t = ph['P0']
        rows.append(
            dict(
                seed=seed,
                system='nested head only (no SCM evidence, confidence)',
                **evaluate(P0t, P0t.max(1), yte, threshold_for_coverage(P0c.max(1), TARGET_COVERAGE)),
            )
        )
        sel_rows.append(
            dict(
                seed=seed,
                head=chosen,
                beta=ph['beta'],
                T=ph['T1'],
                gate=gate,
                **{f'cvF1 {h}': v for h, v in cv_f1.items()},
                **{f'calAURC {g}': v for g, v in cal_aurc.items()},
                **{f'beta {h}': per_head[h]['beta'] for h in heads},
            )
        )
        print(f'seed {seed}: head={chosen}  beta={ph["beta"]:.2f}  gate={gate}', flush=True)
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'picl_v2_explore_per_seed.csv', index=False)
    pd.DataFrame(sel_rows).to_csv(out / 'picl_v2_explore_selection.csv', index=False)
    agg = df.groupby('system').agg(['mean', 'std']).drop(columns='seed')
    agg.columns = [f'{a}_{b}' for a, b in agg.columns]
    agg = agg.reset_index().sort_values('aurc_mean')
    agg.to_csv(out / 'picl_v2_explore_summary.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 80)
    print(
        agg[
            [
                'system',
                'acc_mean',
                'acc_std',
                'macro_f1_mean',
                'aurc_mean',
                'aurc_std',
                'acc85_matched_mean',
                'acc_deployed_mean',
                'coverage_deployed_mean',
                'ece_mean',
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print(pd.DataFrame(sel_rows)[['seed', 'head', 'beta', 'T', 'gate']].to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 52)))
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    main(a.seeds, Path(a.out))
