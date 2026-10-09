"""Leave-one-source-out training and evaluation.

python experiments/loso.py --seeds 52 53 54 55 56 --fold-groups "IEC TC 10;IEEE DataPort;NCEPR;NE Grid,Fujian;Published cases"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    BASELINE_RATIOS,
    baseline_completion,
    TARGET_COVERAGE,
    _proba6,
    accept_stats,
    accepted_acc_at_coverage,
    aurc,
    baseline_features,
    baseline_models,
    composite_from,
    set_seed,
    temperature_apply,
    temperature_fit,
    threshold_for_coverage,
)
from picl.classifier_head import classifier_posterior
from picl.config import load_config
from picl.data import get_log_stats, load_picl_datasets
from picl.inference import _ed_unit, _es_unit, causal_disablement_and_sufficiency, expected_calibration_error
from picl.trainer import train_picl

FAULTS = ['PD', 'D1', 'D2', 'T1', 'T2', 'T3']


def _load_provenance(cfg, provenance_path):
    df = pd.read_csv(cfg.raw['data']['csv_path'])
    fq = cfg.raw['data'].get('filter_query')
    if fq:
        df = df.query(fq).reset_index(drop=True)
    df = df[df['fault_type'].isin(cfg.raw['data']['fault_types'])].reset_index(drop=True)
    if provenance_path:
        prov = pd.read_csv(provenance_path)
        if not {'sample_id', 'source'} <= set(prov.columns):
            raise SystemExit('provenance CSV needs columns sample_id, source')
        df = df.merge(prov[['sample_id', 'source']], on='sample_id', how='left')
        if df['source'].isna().any():
            raise SystemExit(f"{int(df['source'].isna().sum())} records lack a source in the provenance file")
    if 'source' not in df.columns:
        return df, None
    return df, df['source'].astype(str).to_numpy()


FOLD_GROUPS = None


def _folds(mode, seeds, df, sources, n_splits):
    """Yield (seed, fold_name, split_array, source_override, n_sources)."""
    y_all = df['fault_type'].to_numpy()
    groups_all = df['case_id'].to_numpy() if 'case_id' in df.columns else np.arange(len(df))
    n = len(df)
    if mode == 'loso':
        src_names = sorted(set(sources))
        fold_groups = FOLD_GROUPS if FOLD_GROUPS else [[s] for s in src_names]
        from picl.data import group_stratified_split as _gss

        for seed in seeds:
            for grp in fold_groups:
                held = '+'.join(grp)
                test_mask = np.isin(sources, grp)
                rest = np.where(~test_mask)[0]
                roles = _gss(y_all[rest], groups_all[rest], seed, n_folds=4)
                split = np.full(n, 'test', dtype=object)
                split[rest[roles != 'test']] = 'train'
                split[rest[roles == 'test']] = 'cal'
                train_sources = [s for s in src_names if s not in grp]
                code = {s: i for i, s in enumerate(train_sources)}
                for s_ in grp:
                    code[s_] = len(train_sources)
                yield seed, held, split, np.array([code[s] for s in sources], dtype=np.int64), len(train_sources)
    else:
        from picl.data import group_stratified_split as _gss

        for seed in seeds:
            for i in range(n_splits):
                rs = 1000 * seed + i
                split = _gss(y_all, groups_all, rs)
                if sources is None:
                    yield seed, f'split{i}', split, None, None
                else:
                    names = sorted(set(sources))
                    code = {s: j for j, s in enumerate(names)}
                    yield seed, f'split{i}', split, np.array([code[s] for s in sources], dtype=np.int64), len(names)


def main(seeds, out_dir, provenance, results_root, mode='loso', n_splits=10):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg0 = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
    df, sources = _load_provenance(cfg0, provenance)
    if mode == 'loso' and sources is None:
        raise SystemExit('leave-one-source-out needs a `source` column (or --provenance <csv> with sample_id,source)')
    rows, struct_rows = [], []
    for seed, held, split, src_codes, n_src in _folds(mode, seeds, df, sources, n_splits):
        if True:
            set_seed(seed)
            cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
            cfg.raw['experiment']['seed'] = int(seed)
            cfg.raw['data']['_split_override'] = split
            if src_codes is not None:
                cfg.raw['data']['_source_override'] = src_codes
                cfg.raw['data']['n_sources'] = n_src
            train, cal, test = load_picl_datasets(cfg)
            od = Path(results_root) / mode / f'seed{seed}' / f'fold_{held}'
            bundle, summary = train_picl(cfg, train, cal, test, od)
            lm, ls = get_log_stats()
            tr_i, ca_i, te_i = (
                baseline_completion(cfg, train, d, bundle.graph, bundle.scm, lm, ls) for d in (train, cal, test)
            )
            imode = cfg.raw['inference']['intervention_source']
            yc, yt = ca_i.labels.numpy(), te_i.labels.numpy()
            W_eff = (bundle.graph.final_hard_adjacency() * bundle.graph.weight_matrix()).detach()
            with torch.no_grad():
                Pc = bundle.calibrator.transform(
                    classifier_posterior(bundle.head, ca_i, bundle.graph, bundle.scm, cfg_mode=imode)
                ).numpy()
                Pt = bundle.calibrator.transform(
                    classifier_posterior(bundle.head, te_i, bundle.graph, bundle.scm, cfg_mode=imode)
                ).numpy()
                Edc, Esc = causal_disablement_and_sufficiency(
                    ca_i.gas_values, W_eff, bundle.scm, 6, obs=ca_i.observed_gases(6)
                )
                Edt, Est = causal_disablement_and_sufficiency(
                    te_i.gas_values, W_eff, bundle.scm, 6, obs=te_i.observed_gases(6)
                )
            methods = {
                'PICL': (Pc, Pt),
                'PICL + intervention channels': (
                    composite_from(Pc, _ed_unit(Edc).numpy(), _es_unit(Esc).numpy()),
                    composite_from(Pt, _ed_unit(Edt).numpy(), _es_unit(Est).numpy()),
                ),
            }
            for name, model in baseline_models(seed, include_gas_only=True).items():
                r = BASELINE_RATIOS[name]
                model.fit(baseline_features(tr_i, r), tr_i.labels.numpy())
                pc, pt = _proba6(model, baseline_features(ca_i, r)), _proba6(model, baseline_features(te_i, r))
                T = temperature_fit(pc, yc)
                methods[name] = (temperature_apply(pc, T), temperature_apply(pt, T))
            n_test = len(yt)
            class_mix = {f: int((yt == k).sum()) for k, f in enumerate(FAULTS)}
            for name, (Sc, St) in methods.items():
                cc, pc_ = Sc.max(1), Sc.argmax(1)
                ct, pt_ = St.max(1), St.argmax(1)
                corr = pt_ == yt
                thr = threshold_for_coverage(cc, TARGET_COVERAGE)
                tr_st = accept_stats(ct, corr, thr)
                ma = accepted_acc_at_coverage(ct, corr, TARGET_COVERAGE)
                probs = Pt if name.startswith('PICL') else St
                ece = expected_calibration_error(
                    torch.from_numpy(probs.max(1)), torch.from_numpy(probs.argmax(1) == yt), n_bins=15
                )
                rows.append(
                    dict(
                        seed=seed,
                        held_out=held,
                        n_test=n_test,
                        method=name,
                        acc=float(corr.mean()),
                        macro_f1=float(f1_score(yt, pt_, average='macro', labels=list(range(6)), zero_division=0)),
                        ece=ece,
                        aurc=aurc(ct, corr),
                        transferred_coverage=tr_st['coverage'],
                        transferred_acc=tr_st['accepted_acc'],
                        matched_acc85=ma['accepted_acc'],
                        **{f'n_{k}': v for k, v in class_mix.items()},
                    )
                )
            kept = {(e['src'], e['tgt']) for e in summary['learned_edges']}
            struct_rows.append(
                dict(
                    seed=seed,
                    held_out=held,
                    n_train=len(train.labels),
                    n_discoverable_kept=len(kept),
                    kept_edges=json.dumps(sorted(kept)),
                )
            )
            r = rows[-len(methods)]
            print(
                f"  seed {seed} held-out {held:>12s} (n={n_test}): PICL acc={rows[-len(methods)]['acc']:.4f} "
                f"aurc={rows[-len(methods)]['aurc']:.5f} cov={rows[-len(methods)]['transferred_coverage']:.3f}",
                flush=True,
            )
    df_r = pd.DataFrame(rows)
    tag = 'loso' if mode == 'loso' else 'random_splits'
    df_r.to_csv(out_dir / f'{tag}_per_fold.csv', index=False)
    agg = (
        df_r.groupby(['held_out', 'method'])
        .agg(
            n_test=('n_test', 'first'),
            acc=('acc', 'mean'),
            acc_sd=('acc', 'std'),
            macro_f1=('macro_f1', 'mean'),
            ece=('ece', 'mean'),
            aurc=('aurc', 'mean'),
            transferred_coverage=('transferred_coverage', 'mean'),
            transferred_acc=('transferred_acc', 'mean'),
            matched_acc85=('matched_acc85', 'mean'),
        )
        .reset_index()
    )
    agg.to_csv(out_dir / f'{tag}_summary.csv', index=False)
    w = (
        df_r.groupby(['seed', 'method'])
        .apply(
            lambda d: pd.Series(
                {
                    'acc_micro': np.average(d['acc'], weights=d['n_test']),
                    'aurc_micro': np.average(d['aurc'], weights=d['n_test']),
                    'matched_acc85_micro': np.average(d['matched_acc85'], weights=d['n_test']),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    wagg = w.groupby('method').agg(['mean', 'std']).drop(columns='seed')
    wagg.columns = [f'{a}_{b}' for a, b in wagg.columns]
    wagg.reset_index().to_csv(out_dir / f'{tag}_micro.csv', index=False)
    pd.DataFrame(struct_rows).to_csv(out_dir / f'{tag}_graph_stability.csv', index=False)
    pd.set_option('display.width', 250)
    print('\n=== LOSO: accuracy by held-out source ===')
    print(agg.pivot(index='method', columns='held_out', values='acc').round(4).to_string())
    print('\n=== LOSO: AURC by held-out source ===')
    print(agg.pivot(index='method', columns='held_out', values='aurc').round(5).to_string())
    print('\n=== LOSO: realised coverage under the transferred threshold ===')
    print(agg.pivot(index='method', columns='held_out', values='transferred_coverage').round(3).to_string())
    print('\n=== micro-averaged over folds ===')
    print(wagg.round(4).to_string())
    print(f'\nWritten to {out_dir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--out', default='tables')
    ap.add_argument('--provenance', default=None)
    ap.add_argument('--results-root', default='results')
    ap.add_argument(
        '--mode',
        default='loso',
        choices=['loso', 'random'],
        help="'random' = repeated stratified 60/20/20 re-splits of the pooled data",
    )
    ap.add_argument('--n-splits', type=int, default=10)
    ap.add_argument(
        '--fold-groups', default=None, help='";"-separated held-out groups, sources within a group separated by ","'
    )
    a = ap.parse_args()
    if a.fold_groups:
        FOLD_GROUPS = [g.split(',') for g in a.fold_groups.split(';')]
    main(a.seeds, Path(a.out), a.provenance, a.results_root, a.mode, a.n_splits)
