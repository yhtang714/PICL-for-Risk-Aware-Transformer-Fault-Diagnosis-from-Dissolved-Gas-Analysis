from __future__ import annotations
import json
import logging
import resource
import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import torch
from .augment import counterfactual_augment, impute_training_set
from .config import PICLConfig
from .data import PICLDataset, get_log_stats
from .graph import HybridCausalGraph
from .inference import (
    TemperatureCalibrator,
    composite_scores,
    expected_calibration_error,
    full_evaluation,
    optimise_threshold,
)
from .learn import learn_joint, learn_parameters_only
from .scm import LinearGaussianSCM

log = logging.getLogger('picl.trainer')


@dataclass
class PICLBundle:
    cfg: PICLConfig
    graph: HybridCausalGraph
    scm: LinearGaussianSCM
    calibrator: TemperatureCalibrator
    threshold: float
    head: object = None


def _save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)


def _peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _log_causal_graph(graph: HybridCausalGraph, title: str, phase1_kept_keys=None):
    log.info('-' * 70)
    log.info('%s', title)
    log.info('-' * 70)
    hard = graph.hard_edge_summary()
    log.info('Hard edges (from IEC 60599):')
    for e in hard:
        log.info('  %-4s -> %-5s   w = %+.4f', e['src'], e['tgt'], e['weight'])
    disc = graph.edge_posterior_summary()
    kept = [r for r in disc if r['kept']]
    kept.sort(key=lambda r: -abs(r['weight']))
    if phase1_kept_keys is not None:
        primary = [r for r in kept if (r['src'], r['tgt']) in phase1_kept_keys]
        secondary = [r for r in kept if (r['src'], r['tgt']) not in phase1_kept_keys]
        if primary:
            log.info('Primary learned edges (kept in both Phase 1 and Phase 2b):')
            for r in primary:
                log.info(
                    '  %-4s -> %-5s   w = %+.4f   pi = %.3f   var = %.4f   [%s]',
                    r['src'],
                    r['tgt'],
                    r['weight'],
                    r['post_pi'],
                    r['post_var'],
                    r['kind'],
                )
        if secondary:
            log.info('Secondary learned edges (emerged in Phase 2b only):')
            for r in secondary:
                log.info(
                    '  %-4s -> %-5s   w = %+.4f   pi = %.3f   var = %.4f   [%s]',
                    r['src'],
                    r['tgt'],
                    r['weight'],
                    r['post_pi'],
                    r['post_var'],
                    r['kind'],
                )
        if not primary and (not secondary):
            log.info('Learned discoverable edges: (none)')
    elif kept:
        log.info('Learned discoverable edges:')
        for r in kept:
            log.info(
                '  %-4s -> %-5s   w = %+.4f   pi = %.3f   var = %.4f   [%s]',
                r['src'],
                r['tgt'],
                r['weight'],
                r['post_pi'],
                r['post_var'],
                r['kind'],
            )
    else:
        log.info('Learned discoverable edges: (none)')
    weak = [r for r in disc if not r['kept'] and abs(r['weight']) > 0.1]
    weak.sort(key=lambda r: -abs(r['weight']))
    if weak:
        log.info('Weak-signal edges (|w|>0.10, below keep threshold, for reference):')
        for r in weak[:5]:
            log.info('  %-4s -> %-5s   w = %+.4f   pi = %.3f', r['src'], r['tgt'], r['weight'], r['post_pi'])


def train_picl(
    cfg: PICLConfig, train: PICLDataset, cal: PICLDataset, test: PICLDataset, output_dir: Path
) -> Tuple[PICLBundle, Dict]:
    n_sources = int(cfg.raw['data']['n_sources'])
    n_vars = cfg.n_vars
    timings = {}
    _t = time.perf_counter()
    graph = HybridCausalGraph(cfg)
    scm = LinearGaussianSCM(
        n_vars=n_vars, n_sources=n_sources, init_log_var=float(cfg.raw['model']['noise_log_var_init'])
    )
    scm.initialize_mu_from_data(train.data, train.miss_mask)
    log.info('SCM intercept mu initialised from training means')
    log.info('=' * 70)
    log.info('PHASE 1  joint structure + weight learning on original data')
    log.info('=' * 70)
    phase1_cfg = cfg.raw['training']['phase1']
    res1 = learn_joint(
        cfg,
        graph,
        scm,
        train,
        epochs=int(phase1_cfg['epochs']),
        lr=float(phase1_cfg['lr']),
        exclude_synthetic=False,
        anneal_tau=True,
        phase_name='Phase 1',
    )
    log.info('  final loss = %.4f', res1.final_loss)
    timings['stage1_structure_learning_s'] = time.perf_counter() - _t
    _t = time.perf_counter()
    import copy as _copy

    stage1_graph_state = _copy.deepcopy(graph.state_dict())
    stage1_scm_state = _copy.deepcopy(scm.state_dict())
    _log_causal_graph(graph, 'Causal graph after Phase 1')
    phase1_kept_keys = {(e['src'], e['tgt']) for e in res1.kept_edges}
    log.info('=' * 70)
    log.info('PHASE 2a  imputation + counterfactual augmentation')
    log.info('=' * 70)
    log_mu, log_sd = get_log_stats()
    train_imputed_stage2 = impute_training_set(train, graph, scm, log_mu, log_sd, blind_faults=False)
    n_miss = int(train.miss_mask.sum().item())
    log.info('  imputed %d missing entries using conditional-Gaussian mean', n_miss)
    augmented, synth_counts = counterfactual_augment(
        cfg, train_imputed_stage2, graph, scm, rng_seed=int(cfg.raw['experiment']['seed'])
    )
    n_real = int((~augmented.is_synthetic).sum().item())
    n_synth = int(augmented.is_synthetic.sum().item())
    log.info('  augmented set: %d real + %d synthetic = %d total', n_real, n_synth, n_real + n_synth)
    log.info('  synthetic samples per class: %s', synth_counts)
    timings['stage2a_imputation_augmentation_s'] = time.perf_counter() - _t
    _t = time.perf_counter()
    log.info('=' * 70)
    log.info('PHASE 2b  structure refinement (real samples only)')
    log.info('=' * 70)
    phase2s_cfg = cfg.raw['training']['phase2_structure']
    res2s = learn_joint(
        cfg,
        graph,
        scm,
        augmented,
        epochs=int(phase2s_cfg['epochs']),
        lr=float(phase2s_cfg['lr']),
        exclude_synthetic=True,
        anneal_tau=False,
        phase_name='Phase 2b structure',
        use_ce=False,
        fault_to_gas_pos=True,
    )
    log.info('  final loss = %.4f', res2s.final_loss)
    timings['stage2b_structure_refinement_s'] = time.perf_counter() - _t
    _t = time.perf_counter()
    log.info('=' * 70)
    log.info('PHASE 2b  parameter re-estimation (all samples)')
    log.info('=' * 70)
    phase2p_cfg = cfg.raw['training']['phase2_params']
    res2p = learn_parameters_only(
        cfg,
        graph,
        scm,
        augmented,
        epochs=int(phase2p_cfg['epochs']),
        lr=float(phase2p_cfg['lr']),
        phase_name='Phase 2b params',
    )
    log.info('  final loss = %.4f', res2p.final_loss)
    timings['stage2b_parameter_refinement_s'] = time.perf_counter() - _t
    _t = time.perf_counter()
    _log_causal_graph(graph, 'Causal graph after Phase 2b', phase1_kept_keys=phase1_kept_keys)
    train_imputed_head = impute_training_set(train, graph, scm, log_mu, log_sd, blind_faults=True)
    cal_imputed = impute_training_set(cal, graph, scm, log_mu, log_sd, blind_faults=True)
    test_imputed = impute_training_set(test, graph, scm, log_mu, log_sd, blind_faults=True)
    timings['stage3_label_blind_completion_s'] = time.perf_counter() - _t
    _t = time.perf_counter()
    head = None
    clf_post_cal = clf_post_test = clf_post_train = None
    if cfg.raw.get('classifier_head', {}).get('enabled', False):
        log.info('=' * 70)
        log.info('PHASE 4  non-linear classifier head on SCM + DGA-ratio features')
        log.info('=' * 70)
        from .classifier_head import train_classifier_head, classifier_posterior

        t0 = time.time()
        head = train_classifier_head(cfg, train_imputed_head, graph, scm)
        if getattr(head, 'missing_mode', 'completed') == 'reduced':
            from .classifier_head import set_fit_data

            set_fit_data(head, train_imputed_head, cal_imputed, graph.n_faults)
        _imode = str(cfg.raw['inference'].get('intervention_source', 'onehot'))
        clf_post_cal = classifier_posterior(head, cal_imputed, graph, scm, cfg_mode=_imode)
        clf_post_test = classifier_posterior(head, test_imputed, graph, scm, cfg_mode=_imode)
        clf_post_train = classifier_posterior(head, train_imputed_head, graph, scm, cfg_mode=_imode)
        model_name = cfg.raw['classifier_head'].get('model', 'gradient_boosting')
        ensemble = cfg.raw['classifier_head'].get('ensemble', False)
        tag = f'{model_name}+ensemble' if ensemble else model_name
        log.info('  trained classifier head (%s) in %.1fs', tag, time.time() - t0)
    timings['stage3_classifier_head_s'] = time.perf_counter() - _t
    _t = time.perf_counter()
    log.info('=' * 70)
    log.info('PHASE 3  temperature calibration + gate + evaluation')
    log.info('=' * 70)
    n_samp = int(cfg.raw['inference']['n_graph_samples'])
    calibrator = TemperatureCalibrator()
    fuse = bool(cfg.raw.get('classifier_head', {}).get('evidence_fusion', False))
    if fuse and clf_post_cal is not None:
        from .classifier_head import fit_temperature_and_evidence, scm_evidence

        _rows = np.ones(len(cal_imputed.labels), dtype=bool)
        if getattr(head, 'missing_mode', 'completed') == 'reduced':
            _rows = cal_imputed.observed_gases(graph.n_faults).all(1).numpy()
        T_fit, gamma = fit_temperature_and_evidence(
            clf_post_cal.numpy()[_rows],
            scm_evidence(cal_imputed, graph, scm)[_rows],
            cal_imputed.labels.numpy()[_rows],
            fuse=True,
            fit_temperature=bool(cfg.raw['inference']['temperature_scaling']),
        )
        head.evidence_weight = gamma
        head.temperature_ref = T_fit
        clf_post_cal = classifier_posterior(head, cal_imputed, graph, scm, cfg_mode=_imode)
        clf_post_test = classifier_posterior(head, test_imputed, graph, scm, cfg_mode=_imode)
        clf_post_train = classifier_posterior(head, train_imputed_head, graph, scm, cfg_mode=_imode)
        calibrator.temperature = T_fit
        cal_probs = calibrator.transform(clf_post_cal)
        log.info(
            '  joint fit on calibration split: T = %.4f, evidence weight gamma = %.4f (beta = %.4f)',
            T_fit,
            gamma,
            gamma / T_fit,
        )
    elif cfg.raw['inference']['temperature_scaling'] and clf_post_cal is not None:
        ece_before = expected_calibration_error(
            clf_post_cal.max(dim=1).values,
            clf_post_cal.argmax(dim=1) == cal_imputed.labels,
            n_bins=int(cfg.raw['inference'].get('ece_bins', 15)),
        )
        if head is not None and getattr(head, 'missing_mode', 'completed') == 'reduced':
            _rows = cal_imputed.observed_gases(graph.n_faults).all(1)
            calibrator.fit(clf_post_cal[_rows], cal_imputed.labels[_rows])
            head.temperature_ref = calibrator.temperature
            clf_post_cal = classifier_posterior(head, cal_imputed, graph, scm, cfg_mode=_imode)
            clf_post_test = classifier_posterior(head, test_imputed, graph, scm, cfg_mode=_imode)
            clf_post_train = classifier_posterior(head, train_imputed_head, graph, scm, cfg_mode=_imode)
        else:
            calibrator.fit(clf_post_cal, cal_imputed.labels)
        cal_probs = calibrator.transform(clf_post_cal)
        ece_after = expected_calibration_error(
            cal_probs.max(dim=1).values,
            cal_probs.argmax(dim=1) == cal_imputed.labels,
            n_bins=int(cfg.raw['inference'].get('ece_bins', 15)),
        )
        log.info('  calibration-set ECE %.4f -> %.4f', ece_before, ece_after)
    else:
        cal_probs = clf_post_cal
    log.info('  temperature T = %.4f', calibrator.temperature)
    with torch.no_grad():
        cal_scores = composite_scores(
            cfg,
            cal_imputed.gas_values,
            graph,
            scm,
            n_graph_samples=n_samp,
            sources=cal_imputed.source,
            classifier_posterior=cal_probs,
        )
    cal_conf = cal_scores.scores.max(dim=1).values
    cal_pred = cal_scores.pred
    threshold = optimise_threshold(cfg, cal_conf, cal_pred, cal_imputed.labels)
    log.info(
        '  gamma* = %.4f (mode=%s, target_cov=%.3f)',
        threshold,
        str(cfg.raw['inference'].get('threshold_mode', 'max_cov_acc')),
        float(cfg.raw['inference'].get('target_coverage', 0.90)),
    )
    timings['stage3_calibration_and_threshold_s'] = time.perf_counter() - _t
    _t = time.perf_counter()
    test_report = full_evaluation(cfg, test_imputed, graph, scm, calibrator, threshold, clf_posterior=clf_post_test)
    _ti = time.perf_counter()
    _tmp = impute_training_set(test, graph, scm, log_mu, log_sd, blind_faults=True)
    if head is not None:
        _p = classifier_posterior(head, _tmp, graph, scm, cfg_mode=_imode)
        full_evaluation(cfg, _tmp, graph, scm, calibrator, threshold, clf_posterior=_p)
    timings['inference_ms_per_sample'] = 1000.0 * (time.perf_counter() - _ti) / max(1, int(test.data.shape[0]))
    timings['total_training_s'] = sum(v for k, v in timings.items() if k.endswith('_s'))
    timings['peak_rss_mb'] = _peak_rss_mb()
    cal_report = full_evaluation(cfg, cal_imputed, graph, scm, calibrator, threshold, clf_posterior=clf_post_cal)
    train_report = full_evaluation(
        cfg, train_imputed_head, graph, scm, calibrator, threshold, clf_posterior=clf_post_train
    )
    log.info('  TEST  accuracy (all)      = %.4f', test_report.accuracy_all)
    log.info('  TEST  accuracy (accepted) = %.4f', test_report.accuracy_accepted)
    log.info('  TEST  coverage            = %.4f', test_report.coverage)
    log.info('  TEST  ECE                 = %.4f', test_report.ece)
    bundle = PICLBundle(cfg=cfg, graph=graph, scm=scm, calibrator=calibrator, threshold=threshold, head=head)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'models').mkdir(parents=True, exist_ok=True)
    synth_mask = augmented.is_synthetic
    torch.save(
        {
            'graph_state': graph.state_dict(),
            'scm_state': scm.state_dict(),
            'temperature': calibrator.temperature,
            'threshold': threshold,
            'classifier_head': head,
            'stage1_graph_state': stage1_graph_state,
            'stage1_scm_state': stage1_scm_state,
            'synthetic_gas': augmented.gas_values[synth_mask].clone(),
            'synthetic_labels': augmented.labels[synth_mask].clone(),
            'synthetic_sources': augmented.source[synth_mask].clone(),
            'stage2_real_imputed_gas': train_imputed_stage2.gas_values.clone(),
            'stage2_real_labels': train_imputed_stage2.labels.clone(),
        },
        output_dir / 'models' / 'picl_final_model.pt',
    )

    def _report_dict(rep):
        return {
            'accuracy_all': rep.accuracy_all,
            'accuracy_accepted': rep.accuracy_accepted,
            'coverage': rep.coverage,
            'ece': rep.ece,
            'n_samples': rep.n_samples,
            'n_accepted': rep.n_accepted,
            'per_class': rep.per_class,
        }

    final_disc = graph.edge_posterior_summary()
    final_kept = [r for r in final_disc if r['kept']]
    primary_kept = [r for r in final_kept if (r['src'], r['tgt']) in phase1_kept_keys]
    secondary_kept = [r for r in final_kept if (r['src'], r['tgt']) not in phase1_kept_keys]
    results_summary = {
        'evidence_weight': float(getattr(head, 'evidence_weight', 0.0) or 0.0) if head is not None else 0.0,
        'phase1': {'final_loss': res1.final_loss, 'kept_edges': res1.kept_edges},
        'phase2_structure': {'final_loss': res2s.final_loss, 'kept_edges': res2s.kept_edges},
        'phase2_params': {'final_loss': res2p.final_loss},
        'augmentation': {'n_real': n_real, 'n_synth': n_synth, 'synthetic_per_class': synth_counts},
        'calibration': {'temperature': calibrator.temperature, 'threshold': threshold},
        'hard_edges': graph.hard_edge_summary(),
        'learned_edges': final_kept,
        'primary_learned_edges': primary_kept,
        'secondary_learned_edges': secondary_kept,
        'all_discoverable_edges': final_disc,
        'report': {
            'train': _report_dict(train_report),
            'cal': _report_dict(cal_report),
            'test': _report_dict(test_report),
        },
        'timings': timings,
    }
    _save_json(output_dir / 'results_summary.json', results_summary)
    log.info('Saved model to %s', output_dir / 'models' / 'picl_final_model.pt')
    log.info('Saved summary to %s', output_dir / 'results_summary.json')
    return (bundle, results_summary)
