"""Train PICL on one split.

python train.py --seed 52
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from picl.config import load_config
from picl.data import load_picl_datasets
from picl.trainer import train_picl


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser(description='Train PICL on the multi-source DGA dataset.')
    ap.add_argument('--config', default='config/config.yaml')
    ap.add_argument('--prior', default='config/prior_knowledge.yaml')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format='%(message)s',
    )

    cfg = load_config(args.config, args.prior)
    if args.seed is not None:
        cfg.raw['experiment']['seed'] = int(args.seed)
    seed = int(cfg.raw['experiment']['seed'])
    set_seed(seed)

    out = (
        Path(args.output_dir)
        if args.output_dir
        else Path(cfg.raw['experiment'].get('output_dir', 'results')) / 'seeds' / f'seed_{seed}'
    )

    train, cal, test = load_picl_datasets(cfg)
    bundle, summary = train_picl(cfg, train, cal, test, out)

    rep = summary['report']['test']
    print()
    print(
        f"seed {seed}  acc_all={rep['accuracy_all']:.4f}  "
        f"acc_accepted={rep['accuracy_accepted']:.4f}  "
        f"coverage={rep['coverage']:.4f}  ECE={rep['ece']:.4f}"
    )


if __name__ == '__main__':
    main()
