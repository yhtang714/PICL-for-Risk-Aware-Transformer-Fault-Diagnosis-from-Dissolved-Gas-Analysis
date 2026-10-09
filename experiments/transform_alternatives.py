"""Alternative normalisations of the disablement and sufficiency scores in the composite deferral score.

python experiments/transform_alternatives.py --seeds 52 ... 61
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    EVAL_SEEDS,
    TARGET_COVERAGE,
    accepted_acc_at_coverage,
    aurc,
    cols,
    load_scores,
    threshold_for_coverage,
    accept_stats,
)

W = (0.4, 0.3, 0.3)


def _rank_map(ref):
    ref = np.sort(np.asarray(ref).ravel())
    return lambda x: np.searchsorted(ref, x, side='right') / len(ref)


def _minmax_map(ref):
    lo, hi = float(np.min(ref)), float(np.max(ref))
    return lambda x: np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)


def _z_logistic_map(ref):
    m, s = float(np.mean(ref)), float(np.std(ref) + 1e-9)
    return lambda x: 1.0 / (1.0 + np.exp(-(x - m) / s))


def _isotonic_map(ref_x, ref_correct):
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(ref_x, ref_correct)
    return lambda x: iso.predict(np.asarray(x).ravel()).reshape(np.shape(x))


def build_variants(cal, test):
    y_c = cal['label'].to_numpy()
    Pc, Pt = cols(cal, 'picl_P'), cols(test, 'picl_P')
    Edc, Edt = cal['picl_Ed'].to_numpy(), test['picl_Ed'].to_numpy()
    Esc, Est = cols(cal, 'picl_Es'), cols(test, 'picl_Es')
    Educ, Edut = cal['picl_Edu'].to_numpy(), test['picl_Edu'].to_numpy()
    Esuc, Esut = cols(cal, 'picl_Esu'), cols(test, 'picl_Esu')
    top_c = Pc.argmax(1)
    corr_c = (top_c == y_c).astype(float)
    idx = np.arange(len(y_c))
    out = {}

    def comb(P, ed, es):
        return W[0] * P + W[1] * ed[:, None] + W[2] * es

    out['eq36'] = comb(Pt, Edut, Esut)
    out['raw'] = comb(Pt, Edt, Est)
    out['clipped'] = comb(Pt, np.clip(Edt, 0, 10), np.clip(Est, 0, 1))
    r_ed, r_es = _rank_map(Edc), _rank_map(Esc)
    out['rank'] = comb(Pt, r_ed(Edt), r_es(Est))
    m_ed, m_es = _minmax_map(Edc), _minmax_map(Esc)
    out['minmax'] = comb(Pt, m_ed(Edt), m_es(Est))
    z_ed, z_es = _z_logistic_map(Edc), _z_logistic_map(Esc)
    out['zscore_logistic'] = comb(Pt, z_ed(Edt), z_es(Est))
    i_ed = _isotonic_map(Edc, corr_c)
    i_es = _isotonic_map(Esc[idx, top_c], corr_c)
    out['isotonic'] = comb(Pt, i_ed(Edt), i_es(Est))
    from sklearn.linear_model import LogisticRegression

    fc = np.column_stack([Pc[idx, top_c], Educ, Esuc[idx, top_c]])
    lr = LogisticRegression(max_iter=5000).fit(fc, corr_c.astype(int))
    top_t = Pt.argmax(1)
    ft = np.column_stack([Pt[np.arange(len(top_t)), top_t], Edut, Esut[np.arange(len(top_t)), top_t]])
    learned_conf = lr.predict_proba(ft)[:, 1]
    fcc = lr.predict_proba(fc)[:, 1]
    out['confidence'] = Pt
    cal_scores = {
        'eq36': comb(Pc, Educ, Esuc),
        'raw': comb(Pc, Edc, Esc),
        'clipped': comb(Pc, np.clip(Edc, 0, 10), np.clip(Esc, 0, 1)),
        'rank': comb(Pc, r_ed(Edc), r_es(Esc)),
        'minmax': comb(Pc, m_ed(Edc), m_es(Esc)),
        'zscore_logistic': comb(Pc, z_ed(Edc), z_es(Esc)),
        'isotonic': comb(Pc, i_ed(Edc), i_es(Esc)),
        'confidence': Pc,
    }
    return out, cal_scores, (learned_conf, top_t, fcc, top_c)


def main(seeds, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        cal, test = load_scores(seed, 'cal'), load_scores(seed, 'test')
        y_t, y_c = test['label'].to_numpy(), cal['label'].to_numpy()
        variants, cal_scores, learned = build_variants(cal, test)
        for name, S in variants.items():
            conf, pred = S.max(1), S.argmax(1)
            corr = pred == y_t
            Sc = cal_scores[name]
            thr = threshold_for_coverage(Sc.max(1), TARGET_COVERAGE)
            tr = accept_stats(conf, corr, thr)
            ma = accepted_acc_at_coverage(conf, corr, TARGET_COVERAGE)
            rows.append(
                dict(
                    seed=seed,
                    variant=name,
                    acc_full=float(corr.mean()),
                    aurc=aurc(conf, corr),
                    matched_acc85=ma['accepted_acc'],
                    transferred_coverage=tr['coverage'],
                    transferred_acc=tr['accepted_acc'],
                )
            )
        conf, pred, conf_c, pred_c = learned
        corr = pred == y_t
        thr = threshold_for_coverage(conf_c, TARGET_COVERAGE)
        tr = accept_stats(conf, corr, thr)
        ma = accepted_acc_at_coverage(conf, corr, TARGET_COVERAGE)
        rows.append(
            dict(
                seed=seed,
                variant='learned',
                acc_full=float(corr.mean()),
                aurc=aurc(conf, corr),
                matched_acc85=ma['accepted_acc'],
                transferred_coverage=tr['coverage'],
                transferred_acc=tr['accepted_acc'],
            )
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'transform_alternatives_per_seed.csv', index=False)
    agg = (
        df.groupby('variant')
        .agg(
            acc_full=('acc_full', 'mean'),
            aurc=('aurc', 'mean'),
            aurc_sd=('aurc', 'std'),
            matched_acc85=('matched_acc85', 'mean'),
            matched_acc85_sd=('matched_acc85', 'std'),
            transferred_coverage=('transferred_coverage', 'mean'),
            transferred_acc=('transferred_acc', 'mean'),
            n_seeds=('seed', 'nunique'),
        )
        .reset_index()
        .sort_values('aurc')
    )
    agg.to_csv(out_dir / 'transform_alternatives_summary.csv', index=False)
    conf = df[df.variant == 'confidence'].set_index('seed').aurc
    tests = []
    for v in agg.variant:
        if v == 'confidence':
            continue
        a = df[df.variant == v].set_index('seed').aurc.loc[conf.index]
        t, p = stats.ttest_rel(conf, a)
        tests.append(
            dict(variant=v, aurc=a.mean(), gain_vs_confidence=100 * (conf.mean() - a.mean()) / conf.mean(), t=t, p=p)
        )
    tdf = pd.DataFrame(tests)
    tdf.to_csv(out_dir / 'transform_alternatives_tests.csv', index=False)
    print('\n=== Channel-transform alternatives (mean over seeds) ===')
    print(agg.round(5).to_string(index=False))
    print('\nAURC gain over confidence-only (paired t over seeds):')
    print(tdf.round(4).to_string(index=False))
    print(f'\nWritten to {out_dir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=EVAL_SEEDS)
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    main(a.seeds, Path(a.out))
