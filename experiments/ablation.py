"""Pipeline variants used by the ablation and learning-curve experiments."""

from __future__ import annotations
import argparse
import logging
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from picl.config import load_config
from picl.data import load_picl_datasets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_variant(cfg, train, cal, test, output_dir, label):
    from picl.trainer import train_picl

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'models').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    log = logging.getLogger('ablation')
    log.info('==== Ablation variant: %s ====', label)
    bundle, summary = train_picl(cfg, train, cal, test, output_dir)
    return summary['report']['test']


def variant_full():
    cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
    return (cfg, 'PICL (Full)')


def variant_no_risk_gate():
    cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
    cfg.raw['inference']['target_coverage'] = 1.0
    cfg.raw['inference']['threshold_grid_min'] = 0.0
    cfg.raw['inference']['threshold_grid_max'] = 0.99
    return (cfg, '- Risk Gate (cov forced to 1.0)')


def variant_no_gdata():
    cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
    cfg.prior_raw['plausible_edges'] = []
    from picl.config import _build_edges, PICLConfig

    cfg2 = PICLConfig(raw=cfg.raw, prior_raw=cfg.prior_raw)
    cfg2.fault_names = list(cfg.raw['data']['fault_types'])
    cfg2.gas_names = list(cfg.raw['data']['gas_types'])
    cfg2.var_names = cfg2.fault_names + cfg2.gas_names
    cfg2.var_index = {name: i for i, name in enumerate(cfg2.var_names)}
    cfg2.raw['model']['forbid_fault_to_fault'] = True
    cfg2.raw['model']['forbid_gas_to_gas'] = True
    _build_edges(cfg2)
    moved = []
    keep = []
    name_to_idx = cfg2.var_index
    faults = set(cfg2.fault_names)
    gases = set(cfg2.gas_names)
    for e in cfg2.unknown_edges:
        if e.src in faults and e.tgt in gases:
            from picl.config import EdgeSpec

            cfg2.forbidden_edges.append(EdgeSpec(e.src, e.tgt, 'forbidden', 0.0, 'G_data disabled in ablation'))
            moved.append(e)
        else:
            keep.append(e)
    cfg2.unknown_edges = keep
    return (cfg2, '- G_data (only G_hard, no discoverable edges)')


def variant_no_stage2():
    cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
    cfg.raw['training']['phase2_structure']['epochs'] = 0
    cfg.raw['training']['phase2_params']['epochs'] = 0
    return (cfg, '- Stage 2 (no structure refinement, no param re-estimation)')


def variant_no_aug():
    cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
    cfg.raw['augmentation']['target_size'] = 800
    return (cfg, '+ Counterfactual Augmentation (target_size=800)')


def variant_no_temp_calib():
    cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
    cfg.raw['inference']['temperature_scaling'] = False
    return (cfg, '- Temperature Calibration')


def variant_scm_post_gate():
    """Interventions driven by the SCM's own posterior instead of one-hot."""
    cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
    cfg.raw['inference']['intervention_source'] = 'scm_post'
    return (cfg, '~ E_d/E_s from SCM posterior')


def variant_no_causal_channel():
    """Gate on the calibrated classifier probability alone (w_d = w_s = 0)."""
    cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
    cfg.raw['inference']['gate_weights'] = {'posterior': 1.0, 'disablement': 0.0, 'sufficiency': 0.0}
    return (cfg, '- Causal channel (confidence-only gate)')


def variant_uniform_prior():
    """No hard edges and a uniform prior on every discoverable edge."""
    cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
    from picl.config import PICLConfig, _build_edges

    prior = {'hard_edges': [], 'plausible_edges': []}
    cfg2 = PICLConfig(raw=cfg.raw, prior_raw=prior)
    cfg2.fault_names = list(cfg.raw['data']['fault_types'])
    cfg2.gas_names = list(cfg.raw['data']['gas_types'])
    cfg2.var_names = cfg2.fault_names + cfg2.gas_names
    cfg2.var_index = {n: i for i, n in enumerate(cfg2.var_names)}
    _build_edges(cfg2)
    return (cfg2, '- Physics prior (uniform Beta(1,1), no G_hard)')


def variant_no_scm_evidence():
    """Head on completed gases + log-ratios only; no SCM evidence anywhere in the read-out."""
    cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
    cfg.raw['classifier_head']['feature_blocks'] = ['gas', 'dga']
    cfg.raw['classifier_head']['evidence_fusion'] = False
    return (cfg, '- SCM evidence (features and fusion)')


def variant_no_evidence_features():
    cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
    cfg.raw['classifier_head']['feature_blocks'] = ['gas', 'dga']
    return (cfg, '- SCM evidence as head features (fusion kept)')


def variant_no_evidence_fusion():
    cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
    cfg.raw['classifier_head']['evidence_fusion'] = False
    return (cfg, '- SCM evidence fusion (features kept)')


def variant_intervention_gate():
    """Fused posterior plus the disablement and sufficiency scores in the deferral score."""
    cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
    cfg.raw['inference']['gate_weights'] = {'posterior': 0.4, 'disablement': 0.3, 'sufficiency': 0.3}
    return (cfg, '+ intervention channels in the deferral score')


def variant_no_ratios():
    cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
    cfg.raw['classifier_head']['feature_blocks'] = ['gas', 'loglik']
    return (cfg, '- log-ratio features')


VARIANTS = [
    variant_full,
    variant_no_gdata,
    variant_uniform_prior,
    variant_no_stage2,
    variant_no_scm_evidence,
    variant_no_evidence_features,
    variant_no_evidence_fusion,
    variant_no_ratios,
    variant_no_temp_calib,
    variant_intervention_gate,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--output-root', default='results/ablations')
    ap.add_argument(
        '--variants', nargs='+', default=None, help='run only these variant function names, e.g. variant_uniform_prior'
    )
    ap.add_argument('--append', default=None, help='append rows to this CSV instead of overwriting')
    args = ap.parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_fmt = '%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s'
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(level=logging.WARNING, format=log_fmt)
    results = []
    selected = VARIANTS
    if args.variants:
        by_name = {f.__name__: f for f in VARIANTS}
        missing = [n for n in args.variants if n not in by_name]
        if missing:
            raise SystemExit(f'unknown variant(s): {missing}\navailable: {sorted(by_name)}')
        selected = [by_name[n] for n in args.variants]
    for seed in args.seeds:
        for vf in selected:
            cfg, label = vf()
            cfg.raw['experiment']['seed'] = int(seed)
            set_seed(int(seed))
            train, cal, test = load_picl_datasets(cfg)
            safe = label.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '-')
            out_dir = output_root / f'seed{seed}' / safe
            rep = _run_variant(cfg, train, cal, test, out_dir, label)
            row = {
                'seed': seed,
                'variant': label,
                'accuracy_all': rep['accuracy_all'],
                'accuracy_accepted': rep['accuracy_accepted'],
                'coverage': rep['coverage'],
                'ece': rep['ece'],
                'n_accepted': rep['n_accepted'],
                'n_samples': rep['n_samples'],
            }
            print(
                f"  s{seed} {label:>42s}   acc={row['accuracy_all']:.4f}  acc_acc={row['accuracy_accepted']:.4f}  cov={row['coverage']:.4f}  ece={row['ece']:.4f}",
                flush=True,
            )
            results.append(row)
    df = pd.DataFrame(results)
    out_tab = Path('tables')
    out_tab.mkdir(parents=True, exist_ok=True)
    target = Path(args.append) if args.append else out_tab / 'ablation_summary.csv'
    if args.append and target.exists():
        df = pd.concat([pd.read_csv(target), df], ignore_index=True)
    df.to_csv(target, index=False)
    print('\n=== ABLATION SUMMARY ===')
    print(df.to_string(index=False))
    print(f'\nWrote {out_tab}/ablation_summary.csv')


if __name__ == '__main__':
    main()
