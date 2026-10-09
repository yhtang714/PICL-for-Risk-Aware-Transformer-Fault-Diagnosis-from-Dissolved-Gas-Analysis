from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from .config import PICLConfig


@dataclass
class PICLDataset:
    data: torch.Tensor
    gas_values: torch.Tensor
    labels: torch.Tensor
    source: torch.Tensor
    miss_mask: torch.Tensor
    is_synthetic: torch.Tensor
    log_ppm: Optional[torch.Tensor] = None
    obs_mask: Optional[torch.Tensor] = None

    def observed_gases(self, n_faults: int = 6) -> torch.Tensor:
        if self.obs_mask is not None:
            return self.obs_mask
        return ~self.miss_mask[:, n_faults:]


def _resolve_sources(sub, labels, n_sources, seed, split_name):
    """Per-record source indices from the `source` column of the dataset."""
    if 'source' in sub.columns:
        return sub['source'].to_numpy().astype(np.int64), True
    if n_sources <= 1:
        return np.zeros(len(labels), dtype=np.int64), False
    raise ValueError('the dataset has no `source` column; set data.n_sources: 1 for a single-source run')


def group_stratified_split(y, groups, seed, n_folds=5):
    """60/20/20 split by StratifiedGroupKFold: folds 0-2 train, 3 cal, 4 test
    (fold roles rotate with the seed so that every record is a test record in
    some split)."""
    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.zeros(len(y), dtype=int)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        fold[te] = k
    roles = np.array(['train'] * (n_folds - 2) + ['cal', 'test'], dtype=object)
    roles = np.roll(roles, seed % n_folds)
    return roles[fold]


def _normalise_proportions(gas_raw, miss_mask_gas):
    filled = np.where(np.isnan(gas_raw), 0.0, gas_raw)
    row_sum = filled.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= 0.0, 1.0, row_sum)
    return (gas_raw / row_sum).astype(np.float32)


_LAST_LOG_MU = None
_LAST_LOG_SD = None


def get_log_stats():
    return (_LAST_LOG_MU, _LAST_LOG_SD)


def load_picl_datasets(cfg: PICLConfig):
    csv_path = Path(cfg.raw['data']['csv_path'])
    if not csv_path.exists():
        raise FileNotFoundError(f'Dataset not found: {csv_path.resolve()}')
    df = pd.read_csv(csv_path)
    df_all = df.copy()
    grp = cfg.raw['data'].get('filter_group')
    if grp is not None and 'group' in df.columns:
        df = df[df['group'] == grp].reset_index(drop=True)
    fq = cfg.raw['data'].get('filter_query')
    if fq:
        df = df.query(fq).reset_index(drop=True)
    df = df[df['fault_type'].isin(cfg.raw['data']['fault_types'])].reset_index(drop=True)
    wq = cfg.raw['data'].get('weak_label_query')
    weak = None
    if wq:
        weak = df_all.query(wq)
        weak = weak[weak['fault_type'].isin(cfg.raw['data']['fault_types'])].reset_index(drop=True)
    if 'split' not in df.columns:
        df['split'] = 'train'
    required = {'fault_type', 'split'}.union(cfg.gas_names)
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f'CSV missing columns: {missing_cols}')
    fault_to_idx = {name: i for i, name in enumerate(cfg.fault_names)}
    n_vars = cfg.n_vars
    n_sources = int(cfg.raw['data']['n_sources'])
    seed = int(cfg.raw['experiment']['seed'])
    feature_mode = str(cfg.raw['data'].get('gas_feature_mode', 'proportion'))
    gas_raw_full = df[cfg.gas_names].to_numpy(dtype=np.float64)
    log_ppm_full = np.log1p(np.where(np.isnan(gas_raw_full), 0.0, gas_raw_full)).astype(np.float32)
    log_mu = None
    log_sd = None
    split_col = df['split'].to_numpy().astype(object)
    split_mode = str(cfg.raw['data'].get('split_mode', 'fixed'))
    if split_mode == 'random':
        from sklearn.model_selection import train_test_split

        ss = int(cfg.raw['data'].get('split_seed', cfg.raw['experiment']['seed']))
        y_all = df['fault_type'].to_numpy()
        idx = np.arange(len(df))
        tr_idx, rest = train_test_split(idx, test_size=0.40, random_state=ss, stratify=y_all)
        ca_idx, te_idx = train_test_split(rest, test_size=0.50, random_state=ss, stratify=y_all[rest])
        split_col = np.full(len(df), 'train', dtype=object)
        split_col[ca_idx] = 'cal'
        split_col[te_idx] = 'test'
    elif split_mode == 'random_group':
        ss = int(cfg.raw['data'].get('split_seed', cfg.raw['experiment']['seed']))
        split_col = group_stratified_split(df['fault_type'].to_numpy(), df['case_id'].to_numpy(), ss)
    ov = cfg.raw['data'].get('_split_override')
    if ov is not None:
        split_col = np.asarray(ov, dtype=object)
        if len(split_col) != len(df):
            raise ValueError('split override length mismatch')
    if weak is not None and len(weak) > 0:
        df = pd.concat([df, weak], ignore_index=True)
        split_col = np.concatenate([split_col, np.full(len(weak), 'train', dtype=object)])
        gas_raw_full = df[cfg.gas_names].to_numpy(dtype=np.float64)
        log_ppm_full = np.log1p(np.where(np.isnan(gas_raw_full), 0.0, gas_raw_full)).astype(np.float32)
    src_ov = cfg.raw['data'].get('_source_override')
    if src_ov is not None:
        df = df.copy()
        df['source'] = np.asarray(src_ov)
    if 'source' in df.columns and not pd.api.types.is_integer_dtype(df['source']):
        cats = sorted(df['source'].astype(str).unique().tolist())
        df = df.copy()
        df['source'] = df['source'].astype(str).map({c: i for i, c in enumerate(cats)}).astype(np.int64)
        cfg.raw['data']['_source_names'] = cats
        if cfg.raw['data'].get('_source_override') is None:
            cfg.raw['data']['n_sources'] = len(cats)
            n_sources = len(cats)
    if feature_mode == 'log1p_z':
        tr_rows = split_col == 'train'
        tr_log = np.log1p(gas_raw_full[tr_rows])
        log_mu = np.nanmean(tr_log, axis=0).astype(np.float32)
        log_sd = (np.nanstd(tr_log, axis=0) + 1e-08).astype(np.float32)
    out = {}
    for split_name in ('train', 'cal', 'test'):
        sub_mask = split_col == split_name
        sub = df[sub_mask].reset_index(drop=True)
        n = len(sub)
        if n == 0:
            raise ValueError(f'Empty split: {split_name}')
        labels = sub['fault_type'].map(fault_to_idx).to_numpy(dtype=np.int64)
        gas_raw = sub[cfg.gas_names].to_numpy(dtype=np.float32)
        miss_gas = np.isnan(gas_raw)
        log_ppm = log_ppm_full[sub_mask]
        if feature_mode == 'log1p_z':
            log_sub = np.log1p(np.where(np.isnan(gas_raw), 0.0, gas_raw))
            gas_filled = ((log_sub - log_mu) / log_sd).astype(np.float32)
            gas_filled[miss_gas] = 0.0
        else:
            gas_props = _normalise_proportions(gas_raw, miss_gas)
            gas_filled = np.where(np.isnan(gas_props), 0.0, gas_props).astype(np.float32)
        miss_full = np.zeros((n, n_vars), dtype=bool)
        miss_full[:, cfg.n_faults :] = miss_gas
        fault_onehot = np.zeros((n, cfg.n_faults), dtype=np.float32)
        fault_onehot[np.arange(n), labels] = 1.0
        data_full = np.concatenate([fault_onehot, gas_filled], axis=1).astype(np.float32)
        sources, _real_src = _resolve_sources(sub, labels, n_sources, seed, split_name)
        out[split_name] = PICLDataset(
            data=torch.from_numpy(data_full.copy()),
            gas_values=torch.from_numpy(gas_filled.copy()),
            labels=torch.from_numpy(labels.copy()),
            source=torch.from_numpy(sources.copy()),
            miss_mask=torch.from_numpy(miss_full.copy()),
            is_synthetic=torch.zeros(n, dtype=torch.bool),
            log_ppm=torch.from_numpy(log_ppm.copy()),
        )
    global _LAST_LOG_MU, _LAST_LOG_SD
    _LAST_LOG_MU = torch.from_numpy(log_mu) if log_mu is not None else None
    _LAST_LOG_SD = torch.from_numpy(log_sd) if log_sd is not None else None
    return (out['train'], out['cal'], out['test'])


def concat_datasets(a: PICLDataset, b: PICLDataset) -> PICLDataset:

    def _lp(d):
        if d.log_ppm is not None:
            return d.log_ppm
        return torch.zeros(d.data.shape[0], d.gas_values.shape[1], dtype=d.data.dtype)

    return PICLDataset(
        data=torch.cat([a.data, b.data], dim=0),
        gas_values=torch.cat([a.gas_values, b.gas_values], dim=0),
        labels=torch.cat([a.labels, b.labels], dim=0),
        source=torch.cat([a.source, b.source], dim=0),
        miss_mask=torch.cat([a.miss_mask, b.miss_mask], dim=0),
        is_synthetic=torch.cat([a.is_synthetic, b.is_synthetic], dim=0),
        log_ppm=torch.cat([_lp(a), _lp(b)], dim=0),
        obs_mask=torch.cat([a.observed_gases(), b.observed_gases()], dim=0),
    )


def subset(d: PICLDataset, mask: torch.Tensor) -> PICLDataset:
    return PICLDataset(
        data=d.data[mask],
        gas_values=d.gas_values[mask],
        labels=d.labels[mask],
        source=d.source[mask],
        miss_mask=d.miss_mask[mask],
        is_synthetic=d.is_synthetic[mask],
        log_ppm=d.log_ppm[mask] if d.log_ppm is not None else None,
        obs_mask=d.observed_gases()[mask],
    )
