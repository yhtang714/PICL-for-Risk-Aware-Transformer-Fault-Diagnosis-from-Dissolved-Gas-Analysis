"""Helpers shared by the experiment scripts: loading a trained split, PICL scores, calibrated baselines and selective-classification metrics."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from picl.augment import impute_training_set
from picl.classifier_head import classifier_posterior
from picl.config import load_config
from picl.data import PICLDataset, get_log_stats, load_picl_datasets
from picl.graph import HybridCausalGraph
from picl.inference import TemperatureCalibrator, causal_disablement_and_sufficiency, _ed_unit, _es_unit
from picl.scm import LinearGaussianSCM

FAULTS = ['PD', 'D1', 'D2', 'T1', 'T2', 'T3']
GASES = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
DEFAULT_SEEDS = [52, 53, 54, 55, 56]
EVAL_SEEDS = list(range(52, 62))
TARGET_COVERAGE = 0.85
COVERAGE_GRID = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
COST_RATIOS = [1, 2, 5, 10, 20, 50]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class SeedBundle:
    seed: int
    cfg: object
    graph: HybridCausalGraph
    scm: LinearGaussianSCM
    head: object
    calibrator: TemperatureCalibrator
    threshold: float
    train_raw: PICLDataset
    cal_raw: PICLDataset
    test_raw: PICLDataset
    train: PICLDataset
    cal: PICLDataset
    test: PICLDataset
    log_mu: torch.Tensor
    log_sd: torch.Tensor
    summary: dict

    @property
    def W_eff(self):
        with torch.no_grad():
            return (self.graph.final_hard_adjacency() * self.graph.weight_matrix()).detach()


def complete_knn(train_raw: PICLDataset, ds: PICLDataset, log_mu, log_sd, k: int = 5, nf: int = 6) -> PICLDataset:
    """Complete the missing gases of ds by k-nearest neighbours fitted on the training split
    (z-scored log concentrations); the measured-gas mask is kept, so PICL's read-out ignores the
    completed values."""
    from sklearn.impute import KNNImputer

    def nan_arr(d):
        X = d.gas_values.numpy().astype(float).copy()
        X[d.miss_mask[:, nf:].numpy()] = np.nan
        return X

    obs = ds.observed_gases(nf).clone()
    mm = ds.miss_mask[:, nf:]
    if not mm.any():
        return PICLDataset(
            data=ds.data.clone(),
            gas_values=ds.gas_values.clone(),
            labels=ds.labels.clone(),
            source=ds.source.clone(),
            miss_mask=torch.zeros_like(ds.miss_mask),
            is_synthetic=ds.is_synthetic.clone(),
            log_ppm=ds.log_ppm.clone(),
            obs_mask=obs,
        )
    filled = KNNImputer(n_neighbors=k).fit(nan_arr(train_raw)).transform(nan_arr(ds))
    g = ds.gas_values.clone()
    g[mm] = torch.from_numpy(filled.astype(np.float32))[mm]
    d = ds.data.clone()
    d[:, nf:] = g
    lp = ds.log_ppm.clone()
    rec = g * log_sd.unsqueeze(0) + log_mu.unsqueeze(0)
    lp[mm] = rec[mm]
    return PICLDataset(
        data=d,
        gas_values=g,
        labels=ds.labels.clone(),
        source=ds.source.clone(),
        miss_mask=torch.zeros_like(ds.miss_mask),
        is_synthetic=ds.is_synthetic.clone(),
        log_ppm=lp,
        obs_mask=obs,
    )


def baseline_completion(cfg, train_raw: PICLDataset, ds: PICLDataset, graph, scm, log_mu, log_sd) -> PICLDataset:
    """Completed inputs for the baselines (kNN completion)."""
    return impute_training_set(ds, graph, scm, log_mu, log_sd, blind_faults=True)


def load_seed(
    seed: int,
    results_root: str = 'results',
    config_path: str = 'config/config.yaml',
    prior_path: str = 'config/prior_knowledge.yaml',
    cfg_override=None,
) -> SeedBundle:
    cfg = load_config(config_path, prior_path)
    cfg.raw['experiment']['seed'] = int(seed)
    if cfg_override is not None:
        cfg_override(cfg)
    set_seed(seed)
    train, cal, test = load_picl_datasets(cfg)
    g = HybridCausalGraph(cfg)
    s = LinearGaussianSCM(
        n_vars=cfg.n_vars,
        n_sources=int(cfg.raw['data']['n_sources']),
        init_log_var=float(cfg.raw['model']['noise_log_var_init']),
    )
    ck = Path(results_root) / 'seeds' / f'seed_{seed}' / 'models' / 'picl_final_model.pt'
    st = torch.load(ck, map_location='cpu', weights_only=False)
    g.load_state_dict(st['graph_state'])
    s.load_state_dict(st['scm_state'])
    cal_ = TemperatureCalibrator()
    cal_.temperature = float(st['temperature'])
    head = st.get('classifier_head')
    lm, ls = get_log_stats()
    tr_i = impute_training_set(train, g, s, lm, ls, blind_faults=True)
    ca_i = impute_training_set(cal, g, s, lm, ls, blind_faults=True)
    te_i = impute_training_set(test, g, s, lm, ls, blind_faults=True)
    summ_path = Path(results_root) / 'seeds' / f'seed_{seed}' / 'results_summary.json'
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    return SeedBundle(
        seed=seed,
        cfg=cfg,
        graph=g,
        scm=s,
        head=head,
        calibrator=cal_,
        threshold=float(st['threshold']),
        train_raw=train,
        cal_raw=cal,
        test_raw=test,
        train=tr_i,
        cal=ca_i,
        test=te_i,
        log_mu=lm,
        log_sd=ls,
        summary=summary,
    )


def cfg_gate_weights(cfg) -> Tuple[float, float, float]:
    w = cfg.raw['inference']['gate_weights']
    return float(w['posterior']), float(w['disablement']), float(w['sufficiency'])


def picl_channels(
    b: SeedBundle, ds: PICLDataset, weights: Optional[Tuple[float, float, float]] = None
) -> Dict[str, np.ndarray]:
    """Calibrated probability channel, raw and unit-mapped causal channels, composite.
    `weights` defaults to the configured gate weights."""
    if weights is None:
        weights = cfg_gate_weights(b.cfg)
    mode = str(b.cfg.raw['inference'].get('intervention_source', 'onehot'))
    with torch.no_grad():
        P_raw = classifier_posterior(b.head, ds, b.graph, b.scm, cfg_mode=mode)
        P = b.calibrator.transform(P_raw)
        Ed, Es = causal_disablement_and_sufficiency(
            ds.gas_values, b.W_eff, b.scm, b.graph.n_faults, obs=ds.observed_gases(b.graph.n_faults)
        )
        ed_u, es_u = _ed_unit(Ed), _es_unit(Es)
        wp, wd, ws = weights
        S = wp * P + wd * ed_u + ws * es_u
    out = dict(
        P_raw=P_raw.numpy(),
        P=P.numpy(),
        Ed=Ed.numpy(),
        Es=Es.numpy(),
        Ed_unit=ed_u.numpy(),
        Es_unit=es_u.numpy(),
        S=S.numpy(),
        labels=ds.labels.numpy(),
    )
    out['pred'] = out['S'].argmax(1)
    out['conf'] = out['S'].max(1)
    out['pred_clf'] = out['P'].argmax(1)
    out['conf_clf'] = out['P'].max(1)
    return out


def composite_from(P, Ed_unit, Es_unit, weights=(0.4, 0.3, 0.3)):
    wp, wd, ws = weights
    return wp * P + wd * Ed_unit + ws * Es_unit


def temperature_fit(probs_cal: np.ndarray, y_cal: np.ndarray) -> float:
    c = TemperatureCalibrator()
    c.fit(torch.from_numpy(np.asarray(probs_cal, dtype=np.float32)), torch.from_numpy(np.asarray(y_cal)))
    return float(c.temperature)


def temperature_apply(probs: np.ndarray, T: float) -> np.ndarray:
    c = TemperatureCalibrator()
    c.temperature = T
    return c.transform(torch.from_numpy(np.asarray(probs, dtype=np.float32))).numpy()


def baseline_features(ds: PICLDataset, ratios: bool = True) -> np.ndarray:
    """Inputs given to every machine-learning baseline: the five label-blind
    completed gases (log1p, z-scored with training statistics) and, by default,
    the ten pairwise log-ratios used by the IEC/Rogers/Duval ratio methods."""
    from picl.classifier_head import _dga_ratios

    X = ds.gas_values.numpy()
    if ratios:
        X = np.hstack([X, _dga_ratios(ds.log_ppm).numpy()])
    return X.astype(np.float32)


class _BalancedXGB:
    """XGBoost with class-balanced sample weights (sklearn interface)."""

    def __init__(self, seed):
        from xgboost import XGBClassifier

        self.m = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, random_state=seed, n_jobs=2, eval_metric='mlogloss'
        )

    def fit(self, X, y):
        from sklearn.utils.class_weight import compute_sample_weight

        self.m.fit(X, y, sample_weight=compute_sample_weight('balanced', y))
        self.classes_ = self.m.classes_
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


BASELINE_RATIOS = {
    'Random Forest': True,
    'XGBoost': True,
    'SVM': True,
    'ANN': True,
    'Random Forest (gases only)': False,
}


def baseline_models(seed: int, include_gas_only: bool = False):
    """Calibrated discriminative baselines on the gases and their pairwise log-ratios."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    out = {
        'Random Forest': RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=2, class_weight='balanced'),
        'XGBoost': _BalancedXGB(seed),
        'SVM': make_pipeline(
            StandardScaler(), SVC(kernel='rbf', probability=True, random_state=seed, class_weight='balanced')
        ),
        'ANN': make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=seed)
        ),
    }
    if include_gas_only:
        out['Random Forest (gases only)'] = RandomForestClassifier(
            n_estimators=500, random_state=seed, n_jobs=2, class_weight='balanced'
        )
    return out


def _proba6(model, X, K=6):
    P = np.zeros((len(X), K))
    P[:, np.asarray(model.classes_).astype(int)] = model.predict_proba(X)
    return P


def fit_calibrated_baselines(b: SeedBundle, names=None, include_gas_only: bool = True) -> Dict[str, dict]:
    """Train each baseline on the label-blind completed training split, fit a
    temperature on the calibration split, and return probabilities for cal and test."""
    ytr, yca = b.train.labels.numpy(), b.cal.labels.numpy()
    out = {}
    for name, model in baseline_models(b.seed, include_gas_only).items():
        if names is not None and name not in names:
            continue
        r = BASELINE_RATIOS[name]
        Xtr, Xca, Xte = (baseline_features(d, r) for d in (b.train, b.cal, b.test))
        model.fit(Xtr, ytr)
        pc, pt = _proba6(model, Xca), _proba6(model, Xte)
        T = temperature_fit(pc, yca)
        out[name] = dict(
            model=model,
            P_cal_raw=pc,
            P_test_raw=pt,
            T=T,
            P_cal=temperature_apply(pc, T),
            P_test=temperature_apply(pt, T),
        )
    return out


def risk_coverage(conf: np.ndarray, correct: np.ndarray):
    order = np.argsort(-conf, kind='stable')
    c = correct[order].astype(float)
    k = np.arange(1, len(c) + 1)
    risk = np.cumsum(1.0 - c) / k
    return k / len(c), risk, float(np.mean(risk))


def aurc(conf, correct) -> float:
    return risk_coverage(np.asarray(conf), np.asarray(correct))[2]


def threshold_for_coverage(conf_cal: np.ndarray, target: float) -> float:
    """Threshold on the calibration scores that accepts the top `target` fraction."""
    conf_cal = np.asarray(conf_cal)
    if target >= 1.0:
        return -np.inf
    q = np.quantile(conf_cal, 1.0 - target, method='lower')
    return float(q)


def accept_stats(conf: np.ndarray, correct: np.ndarray, thr: float) -> dict:
    acc_mask = np.asarray(conf) >= thr
    n = len(conf)
    n_acc = int(acc_mask.sum())
    correct = np.asarray(correct)
    acc = float(correct[acc_mask].mean()) if n_acc else float('nan')
    n_err = int((~correct[acc_mask]).sum()) if n_acc else 0
    return dict(coverage=n_acc / n, accepted_acc=acc, n_accepted=n_acc, n_errors=n_err, n_deferred=n - n_acc)


def accepted_acc_at_coverage(conf: np.ndarray, correct: np.ndarray, cov: float) -> dict:
    """Accept the top-`cov` fraction of the SAME split (oracle coverage matching)."""
    conf = np.asarray(conf)
    correct = np.asarray(correct)
    n = len(conf)
    k = int(round(cov * n))
    order = np.argsort(-conf, kind='stable')[:k]
    c = correct[order]
    return dict(
        coverage=k / n,
        accepted_acc=float(c.mean()) if k else float('nan'),
        n_accepted=k,
        n_errors=int((~c).sum()) if k else 0,
        n_deferred=n - k,
    )


def min_cost_threshold(conf_cal, correct_cal, r, n_grid=401):
    """Threshold minimising r * accepted errors + deferrals on the calibration split."""
    conf_cal = np.asarray(conf_cal)
    correct_cal = np.asarray(correct_cal)
    grid = np.linspace(conf_cal.min() - 1e-9, conf_cal.max() + 1e-9, n_grid)
    best = (np.inf, grid[0])
    for g in grid:
        m = conf_cal >= g
        cost = r * int((~correct_cal[m]).sum()) + int((~m).sum())
        if cost < best[0]:
            best = (cost, g)
    return float(best[1])


def bootstrap_ci(values: np.ndarray, n_boot=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boots = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_bootstrap_diff(a_correct: np.ndarray, b_correct: np.ndarray, n_boot=2000, seed=0):
    """Bootstrap CI of mean(a) - mean(b) over the same test samples (paired)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a_correct, float)
    b = np.asarray(b_correct, float)
    n = len(a)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d[i] = a[idx].mean() - b[idx].mean()
    return float(np.mean(d)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def dump_seed_scores(
    seed: int, out_dir='results/scores', results_root='results', with_baselines=True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    b = load_seed(seed, results_root=results_root)
    frames = {}
    base = fit_calibrated_baselines(b) if with_baselines else {}
    gamma = float(getattr(b.head, 'evidence_weight', 0.0) or 0.0)
    b.head.evidence_weight = 0.0
    with torch.no_grad():
        Ph_cal = classifier_posterior(b.head, b.cal, b.graph, b.scm).numpy()
        Ph_test = classifier_posterior(b.head, b.test, b.graph, b.scm).numpy()
    b.head.evidence_weight = gamma
    T_head = temperature_fit(Ph_cal, b.cal.labels.numpy())
    Phead = {'cal': temperature_apply(Ph_cal, T_head), 'test': temperature_apply(Ph_test, T_head)}
    Phead_raw = {'cal': Ph_cal, 'test': Ph_test}
    from picl.evidence import dataset_log_posterior

    for split_name, ds in (('cal', b.cal), ('test', b.test)):
        ch = picl_channels(b, ds)
        logq = dataset_log_posterior(ds, b.graph, b.scm)
        df = pd.DataFrame({'label': ch['labels']})
        raw_ds = b.cal_raw if split_name == 'cal' else b.test_raw
        df['n_missing_gas'] = raw_ds.miss_mask[:, b.graph.n_faults :].sum(1).numpy()
        for k in range(6):
            df[f'picl_P{k}'] = ch['P'][:, k]
            df[f'picl_Praw{k}'] = ch['P_raw'][:, k]
            df[f'picl_Es{k}'] = ch['Es'][:, k]
            df[f'picl_Esu{k}'] = ch['Es_unit'][:, k]
            df[f'picl_Phead{k}'] = Phead[split_name][:, k]
            df[f'picl_Phraw{k}'] = Phead_raw[split_name][:, k]
            df[f'picl_logq{k}'] = logq[:, k]
        df['picl_Ed'] = ch['Ed'][:, 0]
        df['picl_Edu'] = ch['Ed_unit'][:, 0]
        for name, r in base.items():
            key = baseline_key(name)
            P = r['P_cal'] if split_name == 'cal' else r['P_test']
            Praw = r['P_cal_raw'] if split_name == 'cal' else r['P_test_raw']
            for k in range(6):
                df[f'{key}_P{k}'] = P[:, k]
                df[f'{key}_Praw{k}'] = Praw[:, k]
        frames[split_name] = df
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    meta = dict(
        seed=seed,
        temperature=b.calibrator.temperature,
        threshold=b.threshold,
        evidence_weight=gamma,
        head_temperature=T_head,
        baseline_temperatures={k: v['T'] for k, v in base.items()},
    )
    for split_name, df in frames.items():
        fn = Path(out_dir) / f'scores_{split_name}.csv'
        old = pd.read_csv(fn) if fn.exists() else None
        df = df.copy()
        df.insert(0, 'seed', seed)
        if old is not None:
            df = pd.concat([old[old.seed != seed], df], ignore_index=True).sort_values('seed', kind='stable')
        df.to_csv(fn, index=False)
    fn = Path(out_dir) / 'meta.json'
    allmeta = json.loads(fn.read_text()) if fn.exists() else {}
    allmeta[str(seed)] = meta
    fn.write_text(json.dumps(allmeta, indent=1))
    _read_scores.cache_clear()
    return frames['cal'], frames['test']


def baseline_key(name: str) -> str:
    return name.replace(' (gases only)', '_gases_only').replace(' ', '_').lower()


@lru_cache(maxsize=4)
def _read_scores(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_scores(seed: int, split: str, score_dir='results/scores') -> pd.DataFrame:
    df = _read_scores(str(Path(score_dir) / f'scores_{split}.csv'))
    return df[df.seed == seed].drop(columns='seed').reset_index(drop=True)


def load_meta(seed: int, score_dir='results/scores') -> dict:
    return json.loads((Path(score_dir) / 'meta.json').read_text())[str(seed)]


def cols(df: pd.DataFrame, prefix: str) -> np.ndarray:
    return df[[f'{prefix}{k}' for k in range(6)]].to_numpy()


def method_scores(df: pd.DataFrame, method: str, weights=(0.4, 0.3, 0.3)) -> Tuple[np.ndarray, np.ndarray]:
    """Return (conf, pred) for a scoring method, e.g. 'picl_composite', 'random_forest' or 'xgboost+causal'; append '_raw' for uncalibrated baselines."""
    Edu = df['picl_Edu'].to_numpy()[:, None]
    Esu = cols(df, 'picl_Esu')
    if method in ('picl', 'picl_confidence'):
        S = cols(df, 'picl_P')
    elif method == 'picl_composite':
        S = composite_from(cols(df, 'picl_P'), Edu, Esu, weights)
    elif method == 'picl_head':
        S = cols(df, 'picl_Phead')
    elif method.endswith('+causal'):
        base = method[: -len('+causal')]
        S = composite_from(cols(df, f'{base}_P'), Edu, Esu, weights)
    elif method.endswith('_raw'):
        S = cols(df, f'{method[:-4]}_Praw')
    else:
        S = cols(df, f'{method}_P')
    return S.max(1), S.argmax(1)


def mean_sd(x):
    x = np.asarray(x, float)
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def dataset_from_ppm(
    ppm: np.ndarray,
    labels: np.ndarray,
    log_mu: torch.Tensor,
    log_sd: torch.Tensor,
    n_faults: int = 6,
    source_idx: int = 0,
) -> PICLDataset:
    """Build a PICLDataset from raw ppm values (NaN = missing) using the training
    split's log statistics, exactly as load_picl_datasets does for the pooled data."""
    ppm = np.asarray(ppm, dtype=np.float64)
    miss = np.isnan(ppm)
    log_sub = np.log1p(np.where(miss, 0.0, ppm))
    lm, ls = log_mu.numpy(), log_sd.numpy()
    gas = ((log_sub - lm) / ls).astype(np.float32)
    gas[miss] = 0.0
    n = len(ppm)
    labels = np.asarray(labels, dtype=np.int64)
    onehot = np.zeros((n, n_faults), dtype=np.float32)
    valid = (labels >= 0) & (labels < n_faults)
    onehot[np.arange(n)[valid], labels[valid]] = 1.0
    data = np.concatenate([onehot, gas], axis=1).astype(np.float32)
    miss_full = np.zeros((n, n_faults + ppm.shape[1]), dtype=bool)
    miss_full[:, n_faults:] = miss
    return PICLDataset(
        data=torch.from_numpy(data),
        gas_values=torch.from_numpy(gas.copy()),
        labels=torch.from_numpy(np.where(valid, labels, 0)),
        source=torch.full((n,), int(source_idx), dtype=torch.long),
        miss_mask=torch.from_numpy(miss_full),
        is_synthetic=torch.zeros(n, dtype=torch.bool),
        log_ppm=torch.from_numpy(log_sub.astype(np.float32)),
    )
