"""Sensitivity of the retained structure and of the diagnosis to the strength of the edge prior.

python experiments/kl_sensitivity.py --seeds 52 53 54 55 56
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import TARGET_COVERAGE, accepted_acc_at_coverage, aurc, set_seed
from picl.augment import impute_training_set
from picl.classifier_head import classifier_posterior
from picl.config import load_config
from picl.data import get_log_stats, load_picl_datasets
from picl.inference import causal_disablement_and_sufficiency
from picl.trainer import train_picl

GAS = {'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6'}
IEC_REF = {
    ('PD', 'H2'),
    ('D1', 'C2H2'),
    ('D1', 'H2'),
    ('D2', 'C2H2'),
    ('D2', 'H2'),
    ('T1', 'CH4'),
    ('T1', 'C2H6'),
    ('T2', 'C2H4'),
    ('T2', 'C2H6'),
    ('T3', 'C2H4'),
    ('T3', 'C2H6'),
}


def main(seeds, weights, out_dir):
    logging.disable(logging.CRITICAL)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for klw in weights:
        for seed in seeds:
            set_seed(seed)
            cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
            cfg.raw['experiment']['seed'] = int(seed)
            cfg.raw['model']['kl_weight'] = float(klw)
            tr, ca, te = load_picl_datasets(cfg)
            od = Path('results') / 'kl_sensitivity' / f'kl{klw}' / f'seed{seed}'
            b, s = train_picl(cfg, tr, ca, te, od)
            ed = s['all_discoverable_edges']
            dev = np.array([abs(e['post_pi'] - e['prior_pi']) for e in ed])
            kept = [(e['src'], e['tgt']) for e in s['learned_edges']]
            fg = {k for k in kept if k[0] not in GAS}
            gg = [k for k in kept if k[0] in GAS]
            shd = len(fg - IEC_REF) + len(IEC_REF - (fg | IEC_REF))
            lm, ls = get_log_stats()
            ca_i = impute_training_set(ca, b.graph, b.scm, lm, ls, blind_faults=True)
            te_i = impute_training_set(te, b.graph, b.scm, lm, ls, blind_faults=True)
            mode = cfg.raw['inference']['intervention_source']
            W_eff = (b.graph.final_hard_adjacency() * b.graph.weight_matrix()).detach()
            with torch.no_grad():
                Pt = b.calibrator.transform(classifier_posterior(b.head, te_i, b.graph, b.scm, cfg_mode=mode)).numpy()
                Edt, Est = causal_disablement_and_sufficiency(te_i.gas_values, W_eff, b.scm, 6)
            St = Pt
            yt = te_i.labels.numpy()
            corr = St.argmax(1) == yt
            rows.append(
                dict(
                    kl_weight=klw,
                    seed=seed,
                    max_post_move=float(dev.max()),
                    mean_post_move=float(dev.mean()),
                    n_fault_gas=len(fg),
                    n_gas_gas=len(gg),
                    shd_fault_gas=shd,
                    acc=float(corr.mean()),
                    aurc=aurc(St.max(1), corr),
                    acc85=accepted_acc_at_coverage(St.max(1), corr, TARGET_COVERAGE)['accepted_acc'],
                    temperature=float(s['calibration']['temperature']),
                    evidence_weight=float(s.get('evidence_weight', 0.0)),
                )
            )
            print(
                f"  kl={klw:>6} seed {seed}: move max {dev.max():.3f} mean {dev.mean():.3f}; f->g {len(fg)} g->g {len(gg)} shd {shd}; acc {corr.mean():.3f} aurc {rows[-1]['aurc']:.3f}",
                flush=True,
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'kl_sensitivity_per_seed.csv', index=False)
    agg = df.groupby('kl_weight').agg(['mean', 'std']).drop(columns='seed')
    agg.columns = [f'{a}_{b}' for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out_dir / 'kl_sensitivity_summary.csv', index=False)
    pd.set_option('display.width', 250)
    print(agg.round(3).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--weights', type=float, nargs='+', default=[0.0, 2.0, 20.0, 200.0, 2000.0])
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    main(a.seeds, a.weights, Path(a.out))
