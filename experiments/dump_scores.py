"""Write per-record scores of PICL and the calibrated baselines for every split to results/scores.

python experiments/dump_scores.py --seeds 52 ... 61
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore', category=RuntimeWarning)

from _common import DEFAULT_SEEDS, dump_seed_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    ap.add_argument('--out', default='results/scores')
    ap.add_argument('--results-root', default='results')
    a = ap.parse_args()
    for s in a.seeds:
        cal, test = dump_seed_scores(s, out_dir=a.out, results_root=a.results_root)
        acc = float((test[[f'picl_P{k}' for k in range(6)]].to_numpy().argmax(1) == test['label']).mean())
        print(f'seed {s}: wrote {len(cal)} cal / {len(test)} test rows; classifier-only test acc={acc:.4f}', flush=True)


if __name__ == '__main__':
    main()
