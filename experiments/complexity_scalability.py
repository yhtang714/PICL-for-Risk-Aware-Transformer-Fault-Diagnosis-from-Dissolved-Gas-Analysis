"""Training and inference cost, and the cost of one objective evaluation as the number of variables grows.

python experiments/complexity_scalability.py --dimension-only --out tables
"""

from __future__ import annotations
import argparse, gc, random, resource, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from picl.augment import impute_training_set
from picl.classifier_head import classifier_posterior, train_classifier_head
from picl.config import load_config
from picl.data import get_log_stats, load_picl_datasets, subset
from picl.graph import HybridCausalGraph
from picl.scm import LinearGaussianSCM
from picl.learn import learn_joint


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def scalability(out_dir, widths=(11, 20, 40, 80, 160, 320), n=638, reps=15, warmup=3):
    """Time one full objective evaluation as p grows, with n fixed."""
    torch.set_num_threads(1)
    rows = []
    for p in widths:
        for r in range(-warmup, reps):
            torch.manual_seed(max(r, 0))
            W = torch.zeros(p, p)
            idx = torch.triu_indices(p, p, offset=1)
            keep = torch.randperm(idx.shape[1])[: min(3 * p, idx.shape[1])]
            W[idx[0, keep], idx[1, keep]] = 0.3
            X = torch.randn(n, p)
            sig = torch.ones(p)
            t0 = time.perf_counter()
            I = torch.eye(p)
            T = torch.linalg.solve(I - W, I)
            Sigma = T.T @ torch.diag(sig) @ T + 1e-6 * I
            L = torch.linalg.cholesky(Sigma)
            z = torch.linalg.solve_triangular(L, X.T, upper=False)
            quad = (z * z).sum(0)
            _ = -0.5 * (quad + 2 * torch.log(torch.diagonal(L)).sum())
            h = torch.trace(torch.matrix_exp(W * W)) - p
            dt = time.perf_counter() - t0
            if r < 0:
                continue
            rows.append(dict(p=p, n=n, rep=r, seconds=dt, n_candidate_edges=p * p - p))
    df = pd.DataFrame(rows)
    agg = (
        df.groupby('p')
        .agg(seconds=('seconds', 'median'), sd=('seconds', 'std'), edges=('n_candidate_edges', 'first'))
        .reset_index()
    )
    base = agg.seconds.iloc[0]
    agg['relative'] = agg.seconds / base
    agg['cubic_prediction'] = (agg.p / agg.p.iloc[0]) ** 3
    agg.to_csv(out_dir / 'scalability_dimension.csv', index=False)
    print('\n=== Cost of one objective evaluation vs number of variables ===')
    print(agg.round(4).to_string(index=False))
    print(
        '`cubic_prediction` is (p/p0)^3, the O(p^3) expectation from the matrix '
        'inverse,\nCholesky and matrix exponential. Compare against `relative`.'
    )
    return agg


def sample_size_and_cost(seeds, out_dir, fracs=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0)):
    rows = []
    for seed in seeds:
        for frac in fracs:
            set_seed(seed)
            cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
            cfg.raw['experiment']['seed'] = int(seed)
            cfg.raw['augmentation']['target_size'] = 1
            cfg.raw['training']['phase1']['epochs'] = 250
            cfg.raw['training']['phase2_structure']['epochs'] = 120
            cfg.raw['training']['phase2_params']['epochs'] = 120
            tr, ca, te = load_picl_datasets(cfg)

            rng = np.random.default_rng(seed)
            lab = tr.labels.numpy()
            keep = np.zeros(len(lab), bool)
            for k in np.unique(lab):
                idx = np.where(lab == k)[0]
                m = max(2, int(round(frac * len(idx))))
                keep[rng.choice(idx, m, replace=False)] = True
            tr_s = subset(tr, torch.from_numpy(keep))

            g = HybridCausalGraph(cfg)
            s = LinearGaussianSCM(
                n_vars=cfg.n_vars,
                n_sources=int(cfg.raw['data']['n_sources']),
                init_log_var=float(cfg.raw['model']['noise_log_var_init']),
            )
            s.initialize_mu_from_data(tr_s.data, tr_s.miss_mask)
            gc.collect()
            m0, t0 = _peak_mb(), time.perf_counter()
            learn_joint(cfg, g, s, tr_s, epochs=250, lr=0.02, exclude_synthetic=False, anneal_tau=True, phase_name='ss')
            fit_s = time.perf_counter() - t0
            lm, ls = get_log_stats()
            tri = impute_training_set(tr_s, g, s, lm, ls, blind_faults=True)
            tei = impute_training_set(te, g, s, lm, ls, blind_faults=True)
            head = train_classifier_head(cfg, tri, g, s)
            t1 = time.perf_counter()
            p = classifier_posterior(head, tei, g, s, cfg_mode=cfg.raw['inference']['intervention_source'])
            infer_ms = 1000.0 * (time.perf_counter() - t1) / len(tei.labels)
            acc = float((p.argmax(1) == tei.labels).float().mean())
            rows.append(
                dict(
                    seed=seed,
                    train_fraction=frac,
                    n_train=int(keep.sum()),
                    accuracy=acc,
                    fit_seconds=fit_s,
                    inference_ms_per_sample=infer_ms,
                    peak_rss_mb=_peak_mb() - m0 + 0.0,
                )
            )
            print(
                f'  seed {seed} frac {frac:.1f} n={int(keep.sum()):4d} ' f'acc={acc:.4f} fit={fit_s:.1f}s', flush=True
            )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'sample_size_and_cost.csv', index=False)
    agg = (
        df.groupby(['train_fraction'])
        .agg(
            n_train=('n_train', 'mean'),
            acc=('accuracy', 'mean'),
            acc_sd=('accuracy', 'std'),
            fit_s=('fit_seconds', 'mean'),
            infer_ms=('inference_ms_per_sample', 'mean'),
        )
        .reset_index()
    )
    full = agg.acc.iloc[-1]
    agg['drop_vs_full'] = agg.acc - full
    print('\n=== Items 6 + 9: training-set size vs accuracy and cost ===')
    print(agg.round(4).to_string(index=False))
    ok = agg[agg.drop_vs_full > -0.02]
    if len(ok):
        print(
            f'\nSmallest training set within 2 pp of the full-data accuracy: '
            f'n = {int(ok.n_train.iloc[0])} ({ok.train_fraction.iloc[0]:.0%} of the '
            f'training split).'
        )
    return agg


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--out', default='tables')
    ap.add_argument('--n', type=int, default=638, help='records per objective evaluation (development set size)')
    ap.add_argument('--dimension-only', action='store_true', help='run only the dimensional scaling sweep')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scalability(out, n=a.n)
    if not a.dimension_only:
        sample_size_and_cost(a.seeds, out)
    print(f'\nWritten to {out}/')
