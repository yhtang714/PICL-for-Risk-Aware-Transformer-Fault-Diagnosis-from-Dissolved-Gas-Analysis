"""Behaviour of the trained models on records without a diagnosed fault.

python experiments/nonfault_screening.py --seeds 52 ... 61
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import FAULTS, TARGET_COVERAGE, dataset_from_ppm, load_seed, picl_channels, threshold_for_coverage
from picl.augment import impute_training_set

G = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']


def main(seeds, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = pd.read_csv('data/dga_provenance.csv')
    nf = prov[prov['group'].isin(['abnormal_no_fault', 'normal'])].reset_index(drop=True)
    x = nf[G].fillna(0.0)
    thc = x['CH4'] + x['C2H2'] + x['C2H4'] + x['C2H6']
    nf['exceeds_attention'] = (x['H2'] > 150) | (x['C2H2'] > 5) | (thc > 150)
    rows, label_rows = [], []
    for seed in seeds:
        b = load_seed(seed)
        ds = dataset_from_ppm(
            nf[G].to_numpy(float),
            np.full(len(nf), -1),
            b.log_mu,
            b.log_sd,
            source_idx=int(b.cfg.raw['data']['n_sources']),
        )
        ds_i = impute_training_set(ds, b.graph, b.scm, b.log_mu, b.log_sd, blind_faults=True)
        ch = picl_channels(b, ds_i)
        ch_te = picl_channels(b, b.test)
        ch_ca = picl_channels(b, b.cal)
        thr_seed = b.threshold
        thr85 = threshold_for_coverage(ch_ca['conf'], TARGET_COVERAGE)
        comp = lambda c: (0.4 * c['P'] + 0.3 * c['Ed_unit'] + 0.3 * c['Es_unit'])[
            np.arange(len(c['P'])), c['P'].argmax(1)
        ]
        thr85_comp = threshold_for_coverage(comp(ch_ca), TARGET_COVERAGE)
        for grp in ('abnormal_no_fault', 'normal', 'all'):
            m = np.ones(len(nf), bool) if grp == 'all' else (nf['group'] == grp).to_numpy()
            if m.sum() == 0:
                continue
            acc_seed = ch['conf'][m] >= thr_seed
            acc85 = ch['conf'][m] >= thr85
            acc_comp = comp(ch)[m] >= thr85_comp
            rows.append(
                dict(
                    seed=seed,
                    group=grp,
                    n=int(m.sum()),
                    exceeds_attention_frac=float(nf.loc[m, 'exceeds_attention'].mean()),
                    accept_rate_seed_thr=float(acc_seed.mean()),
                    accept_rate_85=float(acc85.mean()),
                    accept_rate_85_with_intervention_channels=float(acc_comp.mean()),
                    fault_records_accept_rate_85_with_intervention_channels=float((comp(ch_te) >= thr85_comp).mean()),
                    fault_records_accept_rate_85=float((ch_te['conf'] >= thr85).mean()),
                    mean_max_prob=float(ch['P'].max(1).mean()),
                    mean_max_prob_fault=float(ch_te['P'].max(1).mean()),
                    mean_Ed=float(ch['Ed'][:, 0].mean()),
                    mean_Ed_fault=float(ch_te['Ed'][:, 0].mean()),
                    mean_top_Es=float(ch['Es'].max(1).mean()),
                    mean_top_Es_fault=float(ch_te['Es'].max(1).mean()),
                    mean_composite=float(ch['conf'][m].mean()),
                    mean_composite_fault=float(ch_te['conf'].mean()),
                )
            )
            if grp == 'all':
                for k, f in enumerate(FAULTS):
                    label_rows.append(
                        dict(
                            seed=seed,
                            label=f,
                            frac_predicted=float((ch['pred'] == k).mean()),
                            frac_accepted_with_label=float(((ch['pred'] == k) & acc85).mean()),
                        )
                    )
        print(
            f'  seed {seed}: non-fault accept rate @85 = {rows[-1]["accept_rate_85"]:.3f} (fault records {rows[-1]["fault_records_accept_rate_85"]:.3f})',
            flush=True,
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'nonfault_screening_per_seed.csv', index=False)
    agg = df.groupby('group').agg(['mean', 'std']).drop(columns='seed')
    agg.columns = [f'{a}_{b}' for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out_dir / 'nonfault_screening_summary.csv', index=False)
    dl = pd.DataFrame(label_rows).groupby('label').mean(numeric_only=True).drop(columns='seed').reset_index()
    dl.to_csv(out_dir / 'nonfault_predicted_labels.csv', index=False)
    pd.set_option('display.width', 250)
    print('\n=== Records without a diagnosed fault: gate behaviour (mean over seeds) ===')
    print(
        agg[
            [
                'group',
                'n_mean',
                'exceeds_attention_frac_mean',
                'accept_rate_seed_thr_mean',
                'accept_rate_85_mean',
                'accept_rate_85_with_intervention_channels_mean',
                'fault_records_accept_rate_85_with_intervention_channels_mean',
                'fault_records_accept_rate_85_mean',
                'mean_max_prob_mean',
                'mean_max_prob_fault_mean',
                'mean_Ed_mean',
                'mean_Ed_fault_mean',
                'mean_top_Es_mean',
                'mean_top_Es_fault_mean',
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print('\nlabels assigned to non-fault records (fraction of all; fraction accepted with that label):')
    print(dl.round(3).to_string(index=False))
    print(f'\nWritten to {out_dir}/')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(52, 62)))
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    main(a.seeds, Path(a.out))
