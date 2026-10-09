"""Train PICL on several splits and summarise the results.

python experiments/run_seeds.py --seeds 52 ... 61 --out tables
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from picl.config import load_config
from picl.data import load_picl_datasets
from picl.trainer import train_picl


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--out', default='tables')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format='%(message)s')

    rows = []
    for seed in args.seeds:
        set_seed(seed)
        cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
        cfg.raw['experiment']['seed'] = int(seed)
        train, cal, test = load_picl_datasets(cfg)
        out = Path(f'results/seeds/seed_{seed}')
        bundle, summary = train_picl(cfg, train, cal, test, out)
        rep = summary['report']['test']
        rows.append(
            dict(
                seed=seed,
                acc_all=rep['accuracy_all'],
                acc_accepted=rep['accuracy_accepted'],
                coverage=rep['coverage'],
                ece=rep['ece'],
                temperature=summary['calibration']['temperature'],
                gamma_star=summary['calibration']['threshold'],
            )
        )
        print(
            f"  seed {seed}: acc={rep['accuracy_all']:.4f} "
            f"acc_acc={rep['accuracy_accepted']:.4f} "
            f"cov={rep['coverage']:.4f} ece={rep['ece']:.4f}"
        )

    df = pd.DataFrame(rows)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out) / 'per_seed_summary.csv', index=False)

    print('\n=== PER-SEED SUMMARY ===')
    print(df.to_string(index=False))
    print('\nmean +/- std:')
    for c in ['acc_all', 'acc_accepted', 'coverage', 'ece', 'temperature', 'gamma_star']:
        print(f'  {c:>14s}: {df[c].mean():.4f} +/- {df[c].std():.4f}')
    print(f"\nWrote {args.out}/per_seed_summary.csv")


if __name__ == '__main__':
    main()
