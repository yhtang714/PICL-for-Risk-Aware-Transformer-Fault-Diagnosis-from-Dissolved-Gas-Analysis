"""Align the exported arrays with the provenance records of a split."""

import numpy as np
import pandas as pd
from picl.config import load_config
from picl.data import group_stratified_split


def split_frames(seed, cfg_path='config/config.yaml', prior='config/prior_knowledge.yaml'):
    cfg = load_config(cfg_path, prior)
    df = pd.read_csv(cfg.raw['data']['csv_path'])
    grp = cfg.raw['data'].get('filter_group')
    if grp is not None and 'group' in df.columns:
        df = df[df['group'] == grp].reset_index(drop=True)
    fq = cfg.raw['data'].get('filter_query')
    if fq:
        df = df.query(fq).reset_index(drop=True)
    df = df[df['fault_type'].isin(cfg.raw['data']['fault_types'])].reset_index(drop=True)
    assert cfg.raw['data'].get('split_mode') == 'random_group'
    split = group_stratified_split(df['fault_type'].to_numpy(), df['case_id'].to_numpy(), seed)
    out = {}
    for name, key in (('tr', 'train'), ('ca', 'cal'), ('te', 'test')):
        out[name] = df[split == key].reset_index(drop=True)
    return out


def check(seed, d, frames, gases=('H2', 'CH4', 'C2H2', 'C2H4', 'C2H6')):
    """Verify that the frames line up with the arrays (log1p ppm of measured gases)."""
    for s in ('tr', 'ca', 'te'):
        f = frames[s]
        lp = np.log1p(f[list(gases)].to_numpy(float))
        obs = d[f'{s}_obs']
        a = np.where(obs, d[f'{s}_lp'], 0.0)
        b = np.where(obs, np.nan_to_num(lp, nan=0.0), 0.0)
        assert len(f) == len(d[f'{s}_y']) and np.allclose(a, b, atol=1e-4), (seed, s)
    return True
