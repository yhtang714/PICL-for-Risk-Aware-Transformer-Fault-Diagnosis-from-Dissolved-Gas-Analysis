"""ERH, SHD and prior-posterior agreement of the learned graphs against the IEC 60599 reference.

python experiments/knowledge_alignment.py --seeds 52 ... 61
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from picl.config import load_config
from picl.data import load_picl_datasets
from picl.graph import HybridCausalGraph
from picl.scm import LinearGaussianSCM

N_F, N_G = 6, 5


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _reference_edges(cfg):
    """The IEC reference graph: the 11 hard edges."""
    return {(e.src, e.tgt) for e in cfg.hard_edges}


def _shd(learned, reference, all_nodes):
    """Structural Hamming distance, counting reversals once."""
    add = len(learned - reference)
    dele = len(reference - learned)
    rev = len({(u, v) for (u, v) in learned if (v, u) in reference})
    return add + dele - rev, add, dele, rev


def _forbid_mask(search_space):
    n = N_F + N_G
    m = np.zeros((n, n), dtype=np.float32)
    m[:N_F, N_F:] = 1.0
    if search_space == 'unrestricted':
        m[N_F:, N_F:] = 1.0
        np.fill_diagonal(m, 0.0)
    return m


def _notears_masked(X, mask, lambda_l1=0.01, rho=1.0, epochs=800, lr=0.02, seed=0):
    torch.manual_seed(seed)
    n, d = X.shape
    Xt = torch.from_numpy(X.astype(np.float32))
    M = torch.from_numpy(mask)
    W = torch.zeros(d, d, requires_grad=True)
    opt = torch.optim.Adam([W], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        Wm = W * M
        resid = Xt - Xt @ Wm
        loss = 0.5 / n * (resid**2).sum() + lambda_l1 * Wm.abs().sum()
        h = torch.trace(torch.matrix_exp(Wm * Wm)) - d
        (loss + 0.5 * rho * h * h).backward()
        opt.step()
    return (W.detach() * M).numpy()


def _edges_from_W(W, names, thresh=0.1):
    out = set()
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if i != j and abs(W[i, j]) > thresh:
                out.add((names[i], names[j]))
    return out


def main(seeds, search_space, out_dir):
    rows = []
    for seed in seeds:
        set_seed(seed)
        cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
        cfg.raw['experiment']['seed'] = int(seed)
        train, cal, test = load_picl_datasets(cfg)
        names = cfg.var_names
        ref = _reference_edges(cfg)

        graph = HybridCausalGraph(cfg)
        scm = LinearGaussianSCM(
            n_vars=cfg.n_vars,
            n_sources=int(cfg.raw['data']['n_sources']),
            init_log_var=float(cfg.raw['model']['noise_log_var_init']),
        )
        ck = Path(f'results/seeds/seed_{seed}/models/picl_final_model.pt')
        if not ck.exists():
            raise SystemExit(f'Missing {ck}. Run experiments/run_seeds.py first.')
        st = torch.load(ck, map_location='cpu', weights_only=False)
        graph.load_state_dict(st['graph_state'])
        scm.load_state_dict(st['scm_state'])

        hard = {(e.src, e.tgt) for e in cfg.hard_edges}
        disc = graph.edge_posterior_summary()
        learned = hard | {(r['src'], r['tgt']) for r in disc if r['kept']}
        erh = len(hard & learned) / len(ref)
        shd, add, dele, rev = _shd(learned, ref, names)
        prior = np.array([r['prior_pi'] for r in disc])
        post = np.array([r['post_pi'] for r in disc])
        ppa = float(stats.spearmanr(prior, post).statistic) if prior.std() > 0 else np.nan
        rows.append(
            dict(
                seed=seed,
                method='PICL',
                search_space='fault_to_gas_only + gas_to_gas',
                ERH=erh,
                SHD=shd,
                additions=add,
                deletions=dele,
                reversals=rev,
                PPA=ppa,
                n_edges=len(learned),
            )
        )

        D = train.data.numpy()
        mask = _forbid_mask(search_space)
        for name, W in (
            ('NOTEARS', _notears_masked(D, mask, seed=seed)),
            (
                'DCDI*',
                _notears_masked(
                    np.vstack(
                        [
                            D[train.source.numpy() == s] - D[train.source.numpy() == s].mean(0, keepdims=True)
                            for s in np.unique(train.source.numpy())
                        ]
                    ),
                    mask,
                    seed=seed,
                ),
            ),
        ):
            le = _edges_from_W(W, names)
            erh_b = len(ref & le) / len(ref)
            shd_b, a_b, d_b, r_b = _shd(le, ref, names)
            pri, pos = [], []
            for e in cfg.plausible_edges + cfg.unknown_edges:
                i, j = cfg.var_index[e.src], cfg.var_index[e.tgt]
                pri.append(e.pi if e.kind == 'plausible' else 0.5)
                pos.append(abs(W[i, j]))
            pri, pos = np.array(pri), np.array(pos)
            ppa_b = float(stats.spearmanr(pri, pos).statistic) if pos.std() > 0 else np.nan
            rows.append(
                dict(
                    seed=seed,
                    method=name,
                    search_space=search_space,
                    ERH=erh_b,
                    SHD=shd_b,
                    additions=a_b,
                    deletions=d_b,
                    reversals=r_b,
                    PPA=ppa_b,
                    n_edges=len(le),
                )
            )
        print(f'  seed {seed} done', flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'knowledge_alignment.csv', index=False)
    agg = (
        df.groupby('method')
        .agg(
            ERH=('ERH', 'mean'),
            ERH_sd=('ERH', 'std'),
            SHD=('SHD', 'mean'),
            SHD_sd=('SHD', 'std'),
            PPA=('PPA', 'mean'),
            PPA_sd=('PPA', 'std'),
            n_edges=('n_edges', 'mean'),
            n=('seed', 'nunique'),
        )
        .reset_index()
    )
    print('\n=== Knowledge alignment with IEC 60599 (matched search space) ===')
    print(agg.round(4).to_string(index=False))
    print('\nERH higher is better; SHD lower is better; PPA higher is better.')
    print(f'\nWritten to {out_dir}/knowledge_alignment.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--dag-search-space', default='fault_to_gas_only', choices=['fault_to_gas_only', 'unrestricted'])
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    main(a.seeds, a.dag_search_space, Path(a.out))
