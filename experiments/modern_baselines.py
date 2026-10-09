"""FT-Transformer, graph network, diffusion oversampling and an in-context transformer as baselines.

python experiments/modern_baselines.py --seeds 52 53 54 55 56
"""

from __future__ import annotations
import argparse, math, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from picl.config import load_config
from picl.data import get_log_stats, load_picl_datasets
from picl.graph import HybridCausalGraph
from picl.scm import LinearGaussianSCM

N_G, N_C = 5, 6


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class FTTransformer(nn.Module):
    def __init__(self, d=64, heads=4, layers=3):
        super().__init__()
        self.tok = nn.Parameter(torch.randn(N_G, d) * 0.02)
        self.bias = nn.Parameter(torch.zeros(N_G, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, N_C))

    def forward(self, x):
        t = x.unsqueeze(-1) * self.tok.unsqueeze(0) + self.bias.unsqueeze(0)
        t = torch.cat([self.cls.expand(x.shape[0], -1, -1), t], 1)
        return self.head(self.enc(t)[:, 0])


class GasGNN(nn.Module):
    def __init__(self, d=48, rounds=3):
        super().__init__()
        self.emb = nn.Linear(1, d)
        self.msg = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d)) for _ in range(rounds)]
        )
        self.upd = nn.ModuleList([nn.GRUCell(d, d) for _ in range(rounds)])
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, N_C))

    def forward(self, x):
        B = x.shape[0]
        h = self.emb(x.unsqueeze(-1))
        for msg, upd in zip(self.msg, self.upd):
            hi = h.unsqueeze(2).expand(B, N_G, N_G, h.shape[-1])
            hj = h.unsqueeze(1).expand(B, N_G, N_G, h.shape[-1])
            m = msg(torch.cat([hi, hj], -1)).mean(2)
            h = upd(m.reshape(-1, h.shape[-1]), h.reshape(-1, h.shape[-1])).reshape(B, N_G, -1)
        return self.head(h.mean(1))


class CondDDPM(nn.Module):
    def __init__(self, d=128, T=200):
        super().__init__()
        self.T = T
        self.register_buffer('beta', torch.linspace(1e-4, 0.02, T))
        self.register_buffer('alpha_bar', torch.cumprod(1 - self.beta, 0))
        self.cls = nn.Embedding(N_C, 32)
        self.net = nn.Sequential(nn.Linear(N_G + 32 + 1, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, N_G))

    def forward(self, x, t, y):
        return self.net(torch.cat([x, self.cls(y), (t.float() / self.T).unsqueeze(-1)], -1))

    @torch.no_grad()
    def sample(self, y):
        x = torch.randn(len(y), N_G)
        for t in reversed(range(self.T)):
            tt = torch.full((len(y),), t, dtype=torch.long)
            eps = self(x, tt, y)
            a, ab = 1 - self.beta[t], self.alpha_bar[t]
            x = (x - (1 - a) / math.sqrt(1 - ab) * eps) / math.sqrt(a)
            if t > 0:
                x = x + math.sqrt(self.beta[t]) * torch.randn_like(x)
        return x


class TabPFNStyle(nn.Module):
    def __init__(self, d=64, heads=4, layers=2):
        super().__init__()
        self.qx = nn.Linear(N_G, d)
        self.sx = nn.Linear(N_G + N_C, d)
        enc = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, N_C)

    def forward(self, xq, xs, ys):
        sup = self.sx(torch.cat([xs, torch.eye(N_C)[ys]], -1)).unsqueeze(0)
        q = self.qx(xq).unsqueeze(1)
        seq = torch.cat([q, sup.expand(len(xq), -1, -1)], 1)
        return self.head(self.enc(seq)[:, 0])


def _train_torch(model, Xtr, ytr, epochs=300, lr=1e-3, bs=128, ctx=None):
    torch.set_num_threads(1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            out = model(Xtr[idx], *ctx) if ctx else model(Xtr[idx])
            lossf(out, ytr[idx]).backward()
            opt.step()
    return model


def main(seeds, out_dir, with_incontext=True):
    rows = []
    for seed in seeds:
        set_seed(seed)
        cfg = load_config('config/config.yaml', 'config/prior_knowledge.yaml')
        cfg.raw['experiment']['seed'] = int(seed)
        tr, ca, te = load_picl_datasets(cfg)
        g = HybridCausalGraph(cfg)
        s = LinearGaussianSCM(
            n_vars=cfg.n_vars,
            n_sources=int(cfg.raw['data']['n_sources']),
            init_log_var=float(cfg.raw['model']['noise_log_var_init']),
        )
        st = torch.load(f'results/seeds/seed_{seed}/models/picl_final_model.pt', map_location='cpu', weights_only=False)
        g.load_state_dict(st['graph_state'])
        s.load_state_dict(st['scm_state'])
        lm, ls = get_log_stats()
        from _common import baseline_completion

        tri = baseline_completion(cfg, tr, tr, g, s, lm, ls)
        tei = baseline_completion(cfg, tr, te, g, s, lm, ls)
        Xtr, ytr = tri.gas_values, tri.labels
        Xte, yte = tei.gas_values, tei.labels.numpy()

        def record(name, pred):
            rows.append(
                dict(
                    seed=seed,
                    method=name,
                    accuracy=float((pred == yte).mean()),
                    balanced_accuracy=float(balanced_accuracy_score(yte, pred)),
                    macro_f1=float(f1_score(yte, pred, average='macro', labels=list(range(N_C)), zero_division=0)),
                    mcc=float(matthews_corrcoef(yte, pred)),
                )
            )
            print(f'  seed {seed} {name:18s} acc={rows[-1]["accuracy"]:.4f}', flush=True)

        set_seed(seed)
        m = _train_torch(FTTransformer(), Xtr, ytr, epochs=300).eval()
        with torch.no_grad():
            record('FT-Transformer', m(Xte).argmax(1).numpy())

        set_seed(seed)
        m = _train_torch(GasGNN(), Xtr, ytr, epochs=300).eval()
        with torch.no_grad():
            record('GNN', m(Xte).argmax(1).numpy())

        set_seed(seed)
        dd = CondDDPM()
        opt = torch.optim.Adam(dd.parameters(), lr=1e-3)
        for _ in range(400):
            idx = torch.randint(0, len(Xtr), (128,))
            x0, yb = Xtr[idx], ytr[idx]
            t = torch.randint(0, dd.T, (128,))
            ab = dd.alpha_bar[t].unsqueeze(-1)
            eps = torch.randn_like(x0)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            opt.zero_grad()
            ((dd(xt, t, yb) - eps) ** 2).mean().backward()
            opt.step()
        counts = torch.bincount(ytr, minlength=N_C)
        y_gen = torch.cat(
            [torch.full((int(counts.max()) - int(c),), k, dtype=torch.long) for k, c in enumerate(counts)]
        )
        Xg = dd.sample(y_gen)
        from picl.classifier_head import _GAS_PAIRS

        def with_ratios(Z):
            L = Z.numpy() * ls.numpy() + lm.numpy()
            return np.hstack([Z.numpy(), np.column_stack([L[:, i] - L[:, j] for i, j in _GAS_PAIRS])])

        rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=2, class_weight='balanced')
        rf.fit(with_ratios(torch.cat([Xtr, Xg])), torch.cat([ytr, y_gen]).numpy())
        record('Diffusion+RF', rf.predict(with_ratios(Xte)))

        if with_incontext:
            set_seed(seed)
            sub = torch.randperm(len(Xtr))[:128]
            m = TabPFNStyle()
            m = _train_torch(m, Xtr, ytr, epochs=120, bs=64, ctx=(Xtr[sub], ytr[sub])).eval()
            with torch.no_grad():
                record('TabPFN-style', m(Xte, Xtr[sub], ytr[sub]).argmax(1).numpy())

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'modern_baselines.csv', index=False)
    agg = (
        df.groupby('method')
        .agg(
            acc=('accuracy', 'mean'),
            acc_sd=('accuracy', 'std'),
            bal_acc=('balanced_accuracy', 'mean'),
            macro_f1=('macro_f1', 'mean'),
            mcc=('mcc', 'mean'),
            n=('seed', 'nunique'),
        )
        .reset_index()
        .sort_values('acc', ascending=False)
    )
    print('\n=== Modern architecture baselines ===')
    print(agg.round(4).to_string(index=False))
    print(f'\nWritten to {out_dir}/modern_baselines.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--no-incontext', action='store_true')
    ap.add_argument('--out', default='tables')
    a = ap.parse_args()
    main(a.seeds, Path(a.out), with_incontext=not a.no_incontext)
