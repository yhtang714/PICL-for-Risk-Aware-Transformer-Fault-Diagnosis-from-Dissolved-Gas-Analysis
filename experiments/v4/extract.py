"""Train PICL and export the arrays the read-out needs (gases, ratios, SCM parameters, probabilities).

python experiments/v4/extract.py --seeds 52 ... 61
python experiments/v4/extract.py --loso --seeds 52 53 54 55 56
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings('ignore')
from _common import baseline_completion, load_seed, set_seed
from picl.classifier_head import classifier_posterior
from picl.config import load_config
from picl.data import get_log_stats, load_picl_datasets

NF = 6
FOLD_GROUPS = [['IEC TC 10'], ['IEEE DataPort'], ['NCEPR'], ['NE Grid', 'Fujian'], ['Published cases']]


def pack(prefix, raw, comp, out):
    out[f'{prefix}_gas'] = raw.gas_values.numpy().astype(np.float64)
    out[f'{prefix}_obs'] = raw.observed_gases(NF).numpy().astype(bool)
    out[f'{prefix}_lp'] = raw.log_ppm.numpy().astype(np.float64)
    out[f'{prefix}_gas_c'] = comp.gas_values.numpy().astype(np.float64)
    out[f'{prefix}_lp_c'] = comp.log_ppm.numpy().astype(np.float64)
    out[f'{prefix}_src'] = raw.source.numpy().astype(np.int64)
    out[f'{prefix}_y'] = raw.labels.numpy().astype(np.int64)


def scm_params(graph, scm, out):
    with torch.no_grad():
        out['W'] = (graph.final_hard_adjacency() * graph.weight_matrix()).detach().double().numpy()
        out['mu'] = scm.mu.detach().double().numpy()
        out['log_sigma2'] = scm.log_sigma2.detach().double().numpy()


def in_distribution(seeds, out_dir):
    for seed in seeds:
        b = load_seed(seed)
        out = {}
        for name, raw, comp in (('tr', b.train_raw, b.train), ('ca', b.cal_raw, b.cal), ('te', b.test_raw, b.test)):
            pack(name, raw, comp, out)
            with torch.no_grad():
                out[f'{name}_P'] = (
                    b.calibrator.transform(classifier_posterior(b.head, comp, b.graph, b.scm))
                    .numpy()
                    .astype(np.float64)
                )
        scm_params(b.graph, b.scm, out)
        out['log_mu'], out['log_sd'] = b.log_mu.numpy().astype(np.float64), b.log_sd.numpy().astype(np.float64)
        out['T'] = np.array(b.calibrator.temperature)
        out['gamma'] = np.array(float(b.head.evidence_weight))
        np.savez_compressed(out_dir / f'seed_{seed}.npz', **out)
        print(f'seed {seed}: train {len(out["tr_y"])}, cal {len(out["ca_y"])}, test {len(out["te_y"])}', flush=True)


PRIOR = 'config/prior_knowledge.yaml'


def loso(seeds, out_dir, scratch):
    from picl.data import group_stratified_split as gss
    from picl.trainer import train_picl
    from loso import _load_provenance

    cfg0 = load_config('config/config.yaml', PRIOR)
    df, sources = _load_provenance(cfg0, None)
    y_all = df['fault_type'].to_numpy()
    groups_all = df['case_id'].to_numpy()
    src_names = sorted(set(sources))
    for seed in seeds:
        for grp in FOLD_GROUPS:
            held = '+'.join(grp)
            fn = out_dir / f'seed{seed}_{held.replace(" ", "_")}.npz'
            if fn.exists():
                continue
            test_mask = np.isin(sources, grp)
            rest = np.where(~test_mask)[0]
            roles = gss(y_all[rest], groups_all[rest], seed, n_folds=4)
            split = np.full(len(df), 'test', dtype=object)
            split[rest[roles != 'test']] = 'train'
            split[rest[roles == 'test']] = 'cal'
            train_sources = [s for s in src_names if s not in grp]
            code = {s: i for i, s in enumerate(train_sources)}
            for s_ in grp:
                code[s_] = len(train_sources)
            set_seed(seed)
            cfg = load_config('config/config.yaml', PRIOR)
            cfg.raw['experiment']['seed'] = int(seed)
            cfg.raw['data']['_split_override'] = split
            cfg.raw['data']['_source_override'] = np.array([code[s] for s in sources], dtype=np.int64)
            cfg.raw['data']['n_sources'] = len(train_sources)
            train, cal, test = load_picl_datasets(cfg)
            od = scratch / f'seed{seed}_{held.replace(" ", "_")}'
            bundle, _ = train_picl(cfg, train, cal, test, od)
            lm, ls = get_log_stats()
            out = {}
            for name, raw in (('tr', train), ('ca', cal), ('te', test)):
                comp = baseline_completion(cfg, train, raw, bundle.graph, bundle.scm, lm, ls)
                pack(name, raw, comp, out)
                with torch.no_grad():
                    out[f'{name}_P'] = (
                        bundle.calibrator.transform(classifier_posterior(bundle.head, comp, bundle.graph, bundle.scm))
                        .numpy()
                        .astype(np.float64)
                    )
            scm_params(bundle.graph, bundle.scm, out)
            out['log_mu'], out['log_sd'] = lm.numpy().astype(np.float64), ls.numpy().astype(np.float64)
            out['T'] = np.array(bundle.calibrator.temperature)
            out['gamma'] = np.array(float(bundle.head.evidence_weight))
            out['held_out'] = np.array(held)
            out['train_sources'] = np.array(train_sources)
            np.savez_compressed(fn, **out)
            shutil.rmtree(od, ignore_errors=True)
            acc = (out['te_P'].argmax(1) == out['te_y']).mean()
            print(f'seed {seed} held-out {held}: n_test {len(out["te_y"])}, PICL acc {acc:.3f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 52)))
    ap.add_argument('--loso', action='store_true')
    ap.add_argument('--out', default='results/v4')
    ap.add_argument('--prior', default='config/prior_knowledge.yaml')
    a = ap.parse_args()
    PRIOR = a.prior
    out = Path(a.out) / ('loso' if a.loso else 'indist')
    out.mkdir(parents=True, exist_ok=True)
    if a.loso:
        loso(a.seeds, out, Path('/tmp/v4_scratch'))
    else:
        in_distribution(a.seeds, out)
