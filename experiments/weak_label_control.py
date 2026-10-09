"""Add the rule-labelled utility records to the training split and score on the inspection-labelled test records.

python experiments/weak_label_control.py --seeds 52 ... 61
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    TARGET_COVERAGE,
    _proba6,
    accepted_acc_at_coverage,
    aurc,
    baseline_completion,
    baseline_features,
    baseline_models,
    dataset_from_ppm,
    set_seed,
    temperature_apply,
    temperature_fit,
)
from picl.classifier_head import classifier_posterior
from picl.config import load_config
from picl.data import get_log_stats, load_picl_datasets
from picl.inference import causal_disablement_and_sufficiency
from picl.trainer import train_picl

WEAK_QUERY = "group == 'single_fault' and not label_confirmed and fault_record"
FAULTS = ['PD', 'D1', 'D2', 'T1', 'T2', 'T3']


def _metrics(P_cal, P_te, S_cal, S_te, yc, yt):
    pt = S_te.argmax(1)
    corr = pt == yt
    return dict(
        acc=float(corr.mean()),
        bal_acc=float(balanced_accuracy_score(yt, pt)),
        macro_f1=float(f1_score(yt, pt, average='macro', labels=list(range(6)), zero_division=0)),
        aurc=aurc(S_te.max(1), corr),
        acc85=accepted_acc_at_coverage(S_te.max(1), corr, TARGET_COVERAGE)['accepted_acc'],
        acc_head=float((P_te.argmax(1) == yt).mean()),
    )


def main(seeds, out_dir, results_root):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    weak_df_all = pd.read_csv('data/dga_provenance.csv').query(WEAK_QUERY)
    weak_df_all = weak_df_all[weak_df_all['fault_type'].isin(FAULTS)].reset_index(drop=True)
    for seed in seeds:
        for arm in ('confirmed only', 'confirmed + rule-labelled'):
            set_seed(seed)
            cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
            cfg.raw['experiment']['seed'] = int(seed)
            cfg.raw['data']['weak_label_query'] = WEAK_QUERY if arm.endswith('rule-labelled') else None
            train, cal, test = load_picl_datasets(cfg)
            od = Path(results_root) / 'weak_label' / f'seed{seed}' / arm.replace(' ', '_').replace('+', 'plus')
            bundle, summary = train_picl(cfg, train, cal, test, od)
            lm, ls = get_log_stats()
            tr_i, ca_i, te_i = (
                baseline_completion(cfg, train, d, bundle.graph, bundle.scm, lm, ls) for d in (train, cal, test)
            )
            mode = cfg.raw['inference']['intervention_source']
            yc, yt = ca_i.labels.numpy(), te_i.labels.numpy()
            W_eff = (bundle.graph.final_hard_adjacency() * bundle.graph.weight_matrix()).detach()
            with torch.no_grad():
                Pc = bundle.calibrator.transform(
                    classifier_posterior(bundle.head, ca_i, bundle.graph, bundle.scm, cfg_mode=mode)
                ).numpy()
                Pt = bundle.calibrator.transform(
                    classifier_posterior(bundle.head, te_i, bundle.graph, bundle.scm, cfg_mode=mode)
                ).numpy()
                Edc, Esc = causal_disablement_and_sufficiency(
                    ca_i.gas_values, W_eff, bundle.scm, 6, obs=ca_i.observed_gases(6)
                )
                Edt, Est = causal_disablement_and_sufficiency(
                    te_i.gas_values, W_eff, bundle.scm, 6, obs=te_i.observed_gases(6)
                )
            m = _metrics(Pc, Pt, Pc, Pt, yc, yt)
            rows.append(dict(seed=seed, arm=arm, method='PICL', n_train=len(train.labels), **m))
            wk = dataset_from_ppm(
                weak_df_all[['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']].to_numpy(float),
                weak_df_all['fault_type'].map({f: i for i, f in enumerate(FAULTS)}).to_numpy(),
                lm,
                ls,
                source_idx=int(cfg.raw['data']['n_sources']),
            )
            wk_i = baseline_completion(cfg, train, wk, bundle.graph, bundle.scm, lm, ls)
            with torch.no_grad():
                Pw = bundle.calibrator.transform(
                    classifier_posterior(bundle.head, wk_i, bundle.graph, bundle.scm, cfg_mode=mode)
                ).numpy()
            rows[-1]['three_ratio_agreement'] = float((Pw.argmax(1) == wk_i.labels.numpy()).mean())
            Xtr, ytr = baseline_features(tr_i), tr_i.labels.numpy()
            for name, model in baseline_models(seed).items():
                if name not in ('Random Forest', 'XGBoost'):
                    continue
                model.fit(Xtr, ytr)
                pc, pt = _proba6(model, baseline_features(ca_i)), _proba6(model, baseline_features(te_i))
                T = temperature_fit(pc, yc)
                pc, pt = temperature_apply(pc, T), temperature_apply(pt, T)
                m = _metrics(pc, pt, pc, pt, yc, yt)
                pw = _proba6(model, baseline_features(wk_i))
                rows.append(
                    dict(
                        seed=seed,
                        arm=arm,
                        method=name,
                        n_train=len(train.labels),
                        **m,
                        three_ratio_agreement=float((pw.argmax(1) == wk_i.labels.numpy()).mean()),
                    )
                )
            print(
                f"  seed {seed} {arm:>28s}: PICL acc={rows[-3]['acc']:.4f} bal={rows[-3]['bal_acc']:.4f} f1={rows[-3]['macro_f1']:.4f} aurc={rows[-3]['aurc']:.4f}",
                flush=True,
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'weak_label_control_per_seed.csv', index=False)
    agg = df.groupby(['method', 'arm']).agg(['mean', 'std']).drop(columns='seed')
    agg.columns = [f'{a}_{b}' for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out_dir / 'weak_label_control_summary.csv', index=False)
    tests = []
    for m_ in df.method.unique():
        a = df[(df.method == m_) & (df.arm == 'confirmed + rule-labelled')].set_index('seed')
        b = df[(df.method == m_) & (df.arm == 'confirmed only')].set_index('seed').loc[a.index]
        for col in ('acc', 'bal_acc', 'macro_f1', 'aurc', 'acc85'):
            t, p = stats.ttest_rel(a[col], b[col])
            from metrics_full import nb_corrected_t

            _, pc_, _ = nb_corrected_t((a[col] - b[col]).to_numpy())
            tests.append(
                dict(
                    method=m_,
                    metric=col,
                    with_weak=a[col].mean(),
                    without=b[col].mean(),
                    delta=(a[col] - b[col]).mean(),
                    p=p,
                    p_corrected=pc_,
                )
            )
    pd.DataFrame(tests).to_csv(out_dir / 'weak_label_control_tests.csv', index=False)
    pd.set_option('display.width', 250)
    print('\n=== Weak-label control (mean over seeds) ===')
    print(
        agg[
            [
                'method',
                'arm',
                'n_train_mean',
                'acc_mean',
                'bal_acc_mean',
                'macro_f1_mean',
                'aurc_mean',
                'acc85_mean',
                'three_ratio_agreement_mean',
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(pd.DataFrame(tests).round(4).to_string(index=False))
    print(f'\nWritten to {out_dir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--out', default='tables')
    ap.add_argument('--results-root', default='results')
    a = ap.parse_args()
    main(a.seeds, Path(a.out), a.results_root)
