"""Interventional tests of the fitted SCM (cause activation, cause removal, non-cause independence, sufficiency).

python experiments/counterfactual_faithfulness.py --seed-dirs results/seeds/seed_52 ...
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from picl.config import load_config
from picl.data import load_picl_datasets
from picl.graph import HybridCausalGraph
from picl.scm import LinearGaussianSCM

FAULT_NAMES = ['PD', 'D1', 'D2', 'T1', 'T2', 'T3']
GAS_NAMES = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
N_F = 6
N_G = 5


def _load(seed_dir):
    cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
    train, cal, test = load_picl_datasets(cfg)
    n_sources = int(cfg.raw['data']['n_sources'])
    graph = HybridCausalGraph(cfg)
    scm = LinearGaussianSCM(
        n_vars=cfg.n_vars, n_sources=n_sources, init_log_var=float(cfg.raw['model']['noise_log_var_init'])
    )
    ckpt_path = seed_dir / 'models' / 'picl_final_model.pt'
    if not ckpt_path.exists():
        legacy = seed_dir / 'model_and_meta.pt'
        if legacy.exists():
            ckpt_path = legacy
        else:
            raise FileNotFoundError(f'No checkpoint under {seed_dir}. Run:  python train.py --seed <S>')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    graph.load_state_dict(ckpt['graph_state'])
    scm.load_state_dict(ckpt['scm_state'])
    return (cfg, train, cal, test, graph, scm)


def _do_gas(scm, graph, f_vec):
    with torch.no_grad():
        A = graph.final_hard_adjacency()
        W_eff = (A * graph.weight_matrix()).detach()
        return scm.intervene_on_faults(W_eff, f_vec, graph.n_faults)


def _hard_edge_set(cfg):
    out = set()
    for e in cfg.hard_edges:
        fi = FAULT_NAMES.index(e.src)
        gi = GAS_NAMES.index(e.tgt)
        out.add((fi, gi))
    return out


def test1_cause_activation(scm, graph, cfg):
    hard = _hard_edge_set(cfg)
    rows = []
    n_correct, n_total = (0, 0)
    for fi, gi in sorted(hard):
        f1 = torch.zeros(1, N_F)
        f1[0, fi] = 1.0
        f0 = torch.zeros(1, N_F)
        mu_y1 = _do_gas(scm, graph, f1)[0, gi].item()
        mu_y0 = _do_gas(scm, graph, f0)[0, gi].item()
        delta = mu_y1 - mu_y0
        correct = int(delta > 0)
        rows.append(
            {
                'fault': FAULT_NAMES[fi],
                'gas': GAS_NAMES[gi],
                'E[y|do(f=0)]': mu_y0,
                'E[y|do(f=1)]': mu_y1,
                'delta': delta,
                'increases': bool(correct),
            }
        )
        n_correct += correct
        n_total += 1
    return (rows, n_correct / max(1, n_total))


def test2_cause_removal(scm, graph, cfg, test_ds):
    hard = _hard_edge_set(cfg)
    fault_to_hard_gas = {fi: set() for fi in range(N_F)}
    for fi, gi in hard:
        fault_to_hard_gas[fi].add(gi)
    y_test = test_ds.gas_values
    labels = test_ds.labels
    rows = []
    for fi in range(N_F):
        sel = labels == fi
        if sel.sum() == 0:
            continue
        f_off = torch.zeros(int(sel.sum().item()), N_F)
        mu_off = _do_gas(scm, graph, f_off)
        y_sel = y_test[sel]
        disablement = (y_sel - mu_off).abs()
        hard_g = sorted(fault_to_hard_gas[fi])
        nonhard_g = [g for g in range(N_G) if g not in fault_to_hard_gas[fi]]
        mean_hard = disablement[:, hard_g].mean().item() if hard_g else float('nan')
        mean_nonhard = disablement[:, nonhard_g].mean().item() if nonhard_g else float('nan')
        rows.append(
            {
                'fault': FAULT_NAMES[fi],
                'n_samples': int(sel.sum().item()),
                'n_hard_gases': len(hard_g),
                'mean_disablement_hard': mean_hard,
                'mean_disablement_nonhard': mean_nonhard,
                'ratio_hard_over_nonhard': mean_hard / max(mean_nonhard, 1e-06),
            }
        )
    return rows


def test3_noncause_independence(scm, graph, cfg):
    hard = _hard_edge_set(cfg)
    cause_mags = []
    noncause_mags = []
    rows = []
    for fi in range(N_F):
        f1 = torch.zeros(1, N_F)
        f1[0, fi] = 1.0
        f0 = torch.zeros(1, N_F)
        mu_y1 = _do_gas(scm, graph, f1)[0]
        mu_y0 = _do_gas(scm, graph, f0)[0]
        for gi in range(N_G):
            mag = abs((mu_y1[gi] - mu_y0[gi]).item())
            if (fi, gi) in hard:
                cause_mags.append(mag)
            else:
                noncause_mags.append(mag)
            rows.append(
                {
                    'fault': FAULT_NAMES[fi],
                    'gas': GAS_NAMES[gi],
                    'is_hard': (fi, gi) in hard,
                    'do_effect_magnitude': mag,
                }
            )
    mean_cause = float(np.mean(cause_mags)) if cause_mags else float('nan')
    mean_noncause = float(np.mean(noncause_mags)) if noncause_mags else float('nan')
    return (rows, mean_cause, mean_noncause)


def test4_sufficiency_asymmetry(scm, graph, test_ds):
    y = test_ds.gas_values
    labels = test_ds.labels
    B, G = y.shape
    y_norm = torch.norm(y, dim=1, keepdim=True).clamp(min=1e-08)
    Es = torch.zeros(B, N_F)
    with torch.no_grad():
        A = graph.final_hard_adjacency()
        W_eff = (A * graph.weight_matrix()).detach()
        for k in range(N_F):
            f_int = torch.zeros(B, N_F)
            f_int[:, k] = 1.0
            mu_k = scm.intervene_on_faults(W_eff, f_int, graph.n_faults)
            residual = (y - mu_k).norm(dim=1, keepdim=True)
            Es[:, k] = (1.0 - residual / y_norm).squeeze(1)
    top1 = (Es.argmax(dim=1) == labels).float().mean().item()
    per_class = {}
    for k in range(N_F):
        sel = labels == k
        if sel.sum() > 0:
            per_class[FAULT_NAMES[k]] = float((Es[sel].argmax(dim=1) == k).float().mean().item())
        else:
            per_class[FAULT_NAMES[k]] = float('nan')
    return (top1, per_class, Es)


def aggregate_seeds(seed_dirs):
    out = {'test1': [], 'test2': [], 'test3': [], 'test4': []}
    raw = {}
    for seed_dir in seed_dirs:
        seed_dir = Path(seed_dir)
        cfg, train, cal, test, graph, scm = _load(seed_dir)
        rows1, acc1 = test1_cause_activation(scm, graph, cfg)
        rows2 = test2_cause_removal(scm, graph, cfg, test)
        rows3, mean_c, mean_nc = test3_noncause_independence(scm, graph, cfg)
        top1, per_class, Es = test4_sufficiency_asymmetry(scm, graph, test)
        out['test1'].append({'seed_dir': str(seed_dir), 'acc': acc1, 'rows': rows1})
        out['test2'].append({'seed_dir': str(seed_dir), 'rows': rows2})
        out['test3'].append(
            {
                'seed_dir': str(seed_dir),
                'mean_cause': mean_c,
                'mean_noncause': mean_nc,
                'ratio': mean_nc / max(mean_c, 1e-06),
                'rows': rows3,
            }
        )
        out['test4'].append({'seed_dir': str(seed_dir), 'top1': top1, 'per_class': per_class})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--seed-dirs',
        nargs='+',
        default=None,
        help='seed directories to aggregate; defaults to all under results/seeds',
    )
    args = ap.parse_args()
    if args.seed_dirs is None:
        roots = sorted(Path('results/seeds').glob('seed_*'))
    else:
        roots = [Path(s) for s in args.seed_dirs]
    if not roots:
        print('No seed directories found.')
        return
    print(f'Aggregating over {len(roots)} seed(s): {[str(r) for r in roots]}\n')
    out = aggregate_seeds(roots)
    out_tab = Path('tables')
    out_tab.mkdir(parents=True, exist_ok=True)
    edge_to_deltas = {}
    for s in out['test1']:
        for r in s['rows']:
            key = (r['fault'], r['gas'])
            edge_to_deltas.setdefault(key, []).append(r['delta'])
    t1_summary = []
    n_pass = 0
    for (f, g), deltas in sorted(edge_to_deltas.items()):
        arr = np.array(deltas)
        passed = bool((arr > 0).all())
        t1_summary.append(
            {
                'fault': f,
                'gas': g,
                'delta_mean': float(arr.mean()),
                'delta_std': float(arr.std()),
                'n_seeds_passing': int((arr > 0).sum()),
                'n_seeds': len(arr),
                'all_pass': passed,
            }
        )
        n_pass += int(passed)
    t1_df = pd.DataFrame(t1_summary)
    t1_df.to_csv(out_tab / 'counterfactual_test1_cause_activation.csv', index=False)
    fault_to_vals = {}
    for s in out['test2']:
        for r in s['rows']:
            k = r['fault']
            fault_to_vals.setdefault(k, []).append(r)
    t2_summary = []
    for f, runs in sorted(fault_to_vals.items()):
        df = pd.DataFrame(runs)
        t2_summary.append(
            {
                'fault': f,
                'n_samples': int(df['n_samples'].mean()),
                'n_hard_gases': int(df['n_hard_gases'].mean()),
                'mean_disablement_hard': float(df['mean_disablement_hard'].mean()),
                'std_disablement_hard': float(df['mean_disablement_hard'].std()),
                'mean_disablement_nonhard': float(df['mean_disablement_nonhard'].mean()),
                'std_disablement_nonhard': float(df['mean_disablement_nonhard'].std()),
                'ratio_hard_over_nonhard': float(df['ratio_hard_over_nonhard'].mean()),
            }
        )
    t2_df = pd.DataFrame(t2_summary)
    t2_df.to_csv(out_tab / 'counterfactual_test2_cause_removal.csv', index=False)
    means_c = [s['mean_cause'] for s in out['test3']]
    means_nc = [s['mean_noncause'] for s in out['test3']]
    t3_summary = {
        'mean_cause_effect_size': float(np.mean(means_c)),
        'std_cause_effect_size': float(np.std(means_c)),
        'mean_noncause_effect_size': float(np.mean(means_nc)),
        'std_noncause_effect_size': float(np.std(means_nc)),
        'ratio_noncause_over_cause': float(np.mean(means_nc) / max(np.mean(means_c), 1e-06)),
        'n_seeds': len(means_c),
    }
    pd.DataFrame([t3_summary]).to_csv(out_tab / 'counterfactual_test3_noncause_independence.csv', index=False)
    top1s = [s['top1'] for s in out['test4']]
    t4_summary = {
        'mean_top1_sufficiency_match': float(np.mean(top1s)),
        'std_top1_sufficiency_match': float(np.std(top1s)),
        'n_seeds': len(top1s),
        'random_baseline': 1.0 / N_F,
    }
    pd.DataFrame([t4_summary]).to_csv(out_tab / 'counterfactual_test4_sufficiency_top1.csv', index=False)
    _t1 = f"{t1_df['all_pass'].mean():.3f}  ({n_pass}/{len(t1_df)} edges)"
    _t2 = f"{t2_df['ratio_hard_over_nonhard'].mean():.3f}x"
    _t3 = f"{t3_summary['ratio_noncause_over_cause']:.3f}"
    _t4 = f"{t4_summary['mean_top1_sufficiency_match']:.3f} +/- {t4_summary['std_top1_sufficiency_match']:.3f}"
    _t4r = f"{t4_summary['random_baseline']:.3f}"
    summary = pd.DataFrame(
        [
            {
                'test': 'Test 1 (Cause-Activation)',
                'metric': 'fraction of hard edges where do(f=1) increases gas',
                'PICL': _t1,
                'Random Baseline': '0.500',
            },
            {
                'test': 'Test 2 (Cause-Removal)',
                'metric': 'mean |y - E[y|do(f=0)]| on hard-edge gases / on other gases (z-units, avg over faults)',
                'PICL': _t2,
                'Random Baseline': '1.000',
            },
            {
                'test': 'Test 3 (Non-Cause Independence)',
                'metric': 'mean |do| on non-causal pairs / mean |do| on causal pairs',
                'PICL': _t3,
                'Random Baseline': '1.000',
            },
            {
                'test': 'Test 4 (Sufficiency Asymmetry)',
                'metric': 'top-1 argmax-Es matches true label (test set)',
                'PICL': _t4,
                'Random Baseline': _t4r,
            },
        ]
    )
    summary.to_csv(out_tab / 'counterfactual_faithfulness_summary.csv', index=False)
    print('\n--- Test 1: Cause-Activation (per-edge) ---')
    print(t1_df.to_string(index=False))
    print('\n--- Test 2: Cause-Removal ---')
    print(t2_df.to_string(index=False))
    print('\n--- Test 3: Non-Cause Independence ---')
    print(pd.DataFrame([t3_summary]).to_string(index=False))
    print('\n--- Test 4: Sufficiency Asymmetry ---')
    print(pd.DataFrame([t4_summary]).to_string(index=False))
    print('\n=== MASTER SUMMARY ===')
    print(summary.to_string(index=False))
    print(f'\nWrote per-test tables and counterfactual_faithfulness_summary.csv to {out_tab}/')


if __name__ == '__main__':
    main()
