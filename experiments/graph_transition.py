"""Discoverable edges retained, dropped and added between Stage 1 and Stage 2.

python experiments/graph_transition.py --seeds 52 ... 61
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(seeds, out_dir):
    rows, counts = [], []
    for seed in seeds:
        s = json.load(open(f'results/seeds/seed_{seed}/results_summary.json'))
        st1 = {(e['src'], e['tgt']) for e in s['phase1']['kept_edges']}
        st2 = {(e['src'], e['tgt']) for e in s['learned_edges']}
        hard = {(e['src'], e['tgt']) for e in s['hard_edges']}
        for e in sorted(st1 | st2):
            status = 'kept' if e in st1 and e in st2 else 'dropped in Stage 2' if e in st1 else 'added in Stage 2'
            rows.append(dict(seed=seed, src=e[0], tgt=e[1], status=status))
        counts.append(
            dict(
                seed=seed,
                n_hard=len(hard),
                n_stage1_learned=len(st1),
                n_stage2_learned=len(st2),
                kept=len(st1 & st2),
                dropped=len(st1 - st2),
                added=len(st2 - st1),
            )
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'graph_transition_edges.csv', index=False)
    cdf = pd.DataFrame(counts)
    cdf.to_csv(out_dir / 'graph_transition_counts.csv', index=False)
    print('\n=== Stage-1 -> Stage-2 edge transitions, per seed ===')
    print(cdf.to_string(index=False))
    print(
        f'\nmean: kept {cdf.kept.mean():.1f}, dropped {cdf.dropped.mean():.1f}, '
        f'added {cdf.added.mean():.1f} (hard edges {cdf.n_hard.mean():.0f} always retained)'
    )
    print('\n=== Edge stability across seeds ===')
    piv = (
        df.groupby(['src', 'tgt', 'status'])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values(list(df.status.unique())[0], ascending=False)
    )
    print(piv.to_string(index=False))
    print(f'\nWritten to {out_dir}/graph_transition_*.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    main(a.seeds, Path(a.out))
