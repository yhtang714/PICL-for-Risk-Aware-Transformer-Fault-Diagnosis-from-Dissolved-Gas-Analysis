"""Nonlinear (MLP) mechanism and centred log-ratio coordinates compared with the linear-Gaussian SCM.

python experiments/scm_variants.py --seeds 52 53 54 55 56
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from picl.config import load_config
from picl.data import load_picl_datasets

N_F, N_G = 6, 5


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _clr(raw_ppm, medians):
    """Centred log-ratio on the 5-gas composition; NaNs -> TRAINING-split column median first."""
    x = np.array(raw_ppm, dtype=np.float64)
    for j in range(x.shape[1]):
        m = np.isnan(x[:, j])
        if m.any():
            x[m, j] = medians[j]
    x = np.clip(x, 1e-6, None)
    x = x / x.sum(axis=1, keepdims=True)
    lg = np.log(x)
    return (lg - lg.mean(axis=1, keepdims=True)).astype(np.float32)


class MLPMechanism(nn.Module):
    """Additive-noise mechanism: gas_j = f_j(parents) + eps_j."""

    def __init__(self, d_in, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _fit_linear(Xtr, Ytr):
    """Least-squares mechanism: Y ~ X."""
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    beta, *_ = np.linalg.lstsq(A, Ytr, rcond=None)
    return beta


def _pred_linear(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def _fit_mlp(Xtr, Ytr, seed, epochs=400):
    torch.manual_seed(seed)
    xt = torch.from_numpy(Xtr.astype(np.float32))
    yt = torch.from_numpy(Ytr.astype(np.float32))
    preds = []
    models = []
    for j in range(yt.shape[1]):
        m = MLPMechanism(xt.shape[1])
        opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=1e-4)
        for _ in range(epochs):
            opt.zero_grad()
            loss = ((m(xt) - yt[:, j]) ** 2).mean()
            loss.backward()
            opt.step()
        models.append(m)
    return models


def _pred_mlp(models, X):
    xt = torch.from_numpy(X.astype(np.float32))
    with torch.no_grad():
        return np.column_stack([m(xt).numpy() for m in models])


def main(seeds, out_dir):
    from sklearn.ensemble import RandomForestClassifier

    rows = []
    for seed in seeds:
        set_seed(seed)
        cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
        cfg.raw['experiment']['seed'] = int(seed)
        train, cal, test = load_picl_datasets(cfg)

        import pandas as _pd
        from picl.data import group_stratified_split

        df = _pd.read_csv(cfg.raw['data']['csv_path'])
        fq = cfg.raw['data'].get('filter_query')
        if fq:
            df = df.query(fq).reset_index(drop=True)
        df = df[df['fault_type'].isin(cfg.raw['data']['fault_types'])].reset_index(drop=True)
        gas_cols = cfg.gas_names
        if str(cfg.raw['data'].get('split_mode', 'fixed')) == 'random_group':
            ss = int(cfg.raw['data'].get('split_seed', cfg.raw['experiment']['seed']))
            split_col = group_stratified_split(df['fault_type'].to_numpy(), df['case_id'].to_numpy(), ss)
        else:
            split_col = df['split'].to_numpy().astype(object)
        tr_m = np.asarray(split_col) == 'train'
        te_m = np.asarray(split_col) == 'test'
        assert tr_m.sum() == len(train.labels) and te_m.sum() == len(test.labels), 'split reconstruction mismatch'

        tr_med = np.nanmedian(df.loc[tr_m, gas_cols].to_numpy(dtype=float), axis=0)
        spaces = {
            'log1p_z': (train.gas_values.numpy(), test.gas_values.numpy()),
            'clr': (_clr(df.loc[tr_m, gas_cols].to_numpy(), tr_med), _clr(df.loc[te_m, gas_cols].to_numpy(), tr_med)),
        }
        ytr = train.labels.numpy()
        yte = test.labels.numpy()
        Ftr = np.eye(N_F, dtype=np.float32)[ytr]
        Fte = np.eye(N_F, dtype=np.float32)[yte]

        for space, (Gtr, Gte) in spaces.items():
            for mech in ('linear', 'mlp'):
                Xtr = np.hstack([Ftr, Gtr]).astype(np.float32)
                Xte = np.hstack([Fte, Gte]).astype(np.float32)

                def _mask_self(X, j):
                    Z = X.copy()
                    Z[:, N_F + j] = 0.0
                    return Z

                Ptr = np.zeros_like(Gtr)
                Pte = np.zeros_like(Gte)
                betas, mlps = [], []
                for j in range(N_G):
                    a_tr, a_te = _mask_self(Xtr, j), _mask_self(Xte, j)
                    if mech == 'linear':
                        b = _fit_linear(a_tr, Gtr[:, j : j + 1])
                        betas.append(b)
                        Ptr[:, j] = _pred_linear(b, a_tr).ravel()
                        Pte[:, j] = _pred_linear(b, a_te).ravel()
                    else:
                        m = _fit_mlp(a_tr, Gtr[:, j : j + 1], seed)
                        mlps.append(m)
                        Ptr[:, j] = _pred_mlp(m, a_tr).ravel()
                        Pte[:, j] = _pred_mlp(m, a_te).ravel()

                rmse_tr = float(np.sqrt(((Gtr - Ptr) ** 2).mean()))
                rmse_te = float(np.sqrt(((Gte - Pte) ** 2).mean()))

                def feats(G):
                    per_class = []
                    for k in range(N_F):
                        e = np.zeros((len(G), N_F), dtype=np.float32)
                        e[:, k] = 1.0
                        Xk = np.hstack([e, G]).astype(np.float32)
                        mu_k = np.zeros_like(G)
                        for j in range(N_G):
                            Z = Xk.copy()
                            Z[:, N_F + j] = 0.0
                            mu_k[:, j] = (
                                _pred_linear(betas[j], Z).ravel() if mech == 'linear' else _pred_mlp(mlps[j], Z).ravel()
                            )
                        per_class.append(np.linalg.norm(G - mu_k, axis=1, keepdims=True))
                    return np.hstack([G] + per_class)

                clf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=2, class_weight='balanced')
                clf.fit(feats(Gtr), ytr)
                acc = float((clf.predict(feats(Gte)) == yte).mean())

                rows.append(
                    dict(
                        seed=seed,
                        feature_space=space,
                        mechanism=mech,
                        variant=f'{mech}-{space}',
                        train_rmse=rmse_tr,
                        test_rmse=rmse_te,
                        downstream_acc=acc,
                    )
                )
                print(f'  seed {seed}  {mech:6s} / {space:8s}  ' f'test_rmse={rmse_te:.4f}  downstream_acc={acc:.4f}')

    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / 'scm_variants.csv', index=False)

    agg = (
        df_out.groupby(['feature_space', 'mechanism'])
        .agg(
            test_rmse=('test_rmse', 'mean'),
            test_rmse_sd=('test_rmse', 'std'),
            acc=('downstream_acc', 'mean'),
            acc_sd=('downstream_acc', 'std'),
            n=('seed', 'nunique'),
        )
        .reset_index()
    )
    print('\n=== Mechanism / feature-space sensitivity ===')
    print(agg.round(4).to_string(index=False))
    lin = agg[(agg.mechanism == 'linear') & (agg.feature_space == 'log1p_z')].acc.iloc[0]
    best = agg.loc[agg.acc.idxmax()]
    print(f"\nlinear / log1p_z accuracy = {lin:.4f}")
    print(f"best variant = {best.mechanism} / {best.feature_space} = {best.acc:.4f} " f"({best.acc - lin:+.4f})")
    print('\nRMSE is only comparable WITHIN a feature space; accuracy is comparable across.')
    print(f'\nWritten to {out_dir}/scm_variants.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    main(a.seeds, Path(a.out))
