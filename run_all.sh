#!/usr/bin/env bash
# Full reproduction: evaluation splits, confirmation splits, then tables and figures.
set -o pipefail
export OMP_NUM_THREADS=1 PYTHONPATH=.
S="52 53 54 55 56 57 58 59 60 61"
S5="52 53 54 55 56"
run() { echo "=== $1 === $(date +%H:%M:%S)"; shift; "$@" 2>&1 | grep -v Warning | grep -v "warnings.warn"; }
mkdir -p tables results

run run_seeds python3 experiments/run_seeds.py --seeds $S --out tables
run dump_scores python3 experiments/dump_scores.py --seeds $S
run main_table python3 experiments/main_table.py --seeds $S
run metrics_full python3 experiments/metrics_full.py --seeds $S
run matched_coverage python3 experiments/matched_coverage.py --seeds $S
run paired_differences python3 experiments/paired_differences.py --seeds $S
run weight_sensitivity python3 experiments/weight_sensitivity.py --seeds $S
run transform_alternatives python3 experiments/transform_alternatives.py --seeds $S
run knowledge_alignment python3 experiments/knowledge_alignment.py --seeds $S
run graph_transition python3 experiments/graph_transition.py --seeds $S
run counterfactual_faithfulness python3 experiments/counterfactual_faithfulness.py --seed-dirs $(for s in $S; do echo -n "results/seeds/seed_$s "; done)
run residual_diagnostics python3 experiments/residual_diagnostics.py --seeds $S
run intervention_magnitude python3 experiments/intervention_magnitude.py --seeds $S
run nonfault_screening python3 experiments/nonfault_screening.py --seeds $S
run mixed_fault_ppm python3 experiments/mixed_fault_ppm.py --seeds $S
run causal_necessity python3 experiments/causal_necessity.py --seeds $S
run threshold_analysis python3 experiments/threshold_analysis.py --seeds $S
run evidence_variants python3 experiments/picl_v3_explore.py --seeds $S --out tables
run missingness_patterns python3 experiments/missingness_patterns.py --seeds $S
run loso python3 experiments/loso.py --seeds $S5 --fold-groups "IEC TC 10;IEEE DataPort;NCEPR;NE Grid,Fujian;Published cases"
run weak_label_control python3 experiments/weak_label_control.py --seeds $S
run learning_curve python3 experiments/learning_curve.py --seeds $S --out tables
run ablation_matched python3 experiments/ablation_matched.py --seeds $S
run scm_variants python3 experiments/scm_variants.py --seeds $S5
run kl_sensitivity python3 experiments/kl_sensitivity.py --seeds $S5
run modern_baselines python3 experiments/modern_baselines.py --seeds $S5
run misspecification_sweep python3 experiments/misspecification_test.py --reps 40 --out tables
run scalability python3 experiments/complexity_scalability.py --dimension-only --out tables

run extract python3 experiments/v4/extract.py --seeds $S
run extract_loso python3 experiments/v4/extract.py --loso --seeds $S5
run hierarchical python3 experiments/v4/eval_indist.py --seeds $S --out tables
run unseen_source python3 experiments/v4/eval_loso.py --seeds $S5 --out tables
run recalibration env PYTHONPATH=.:experiments/v4 python3 experiments/v5/eval_fewshot.py --seeds $S5 --out tables
run catboost python3 experiments/v6/catboost_eval.py --seeds $S --root results/v4/indist --out tables
run record_tests python3 experiments/v6/indist_record_tests.py --seeds $S
run transformer_tests env PYTHONPATH=.:experiments/v4 python3 experiments/v6/transformer_tests.py
run ratio_labels python3 experiments/v6/ratio_label_sensitivity.py
run loso_record_tests env PYTHONPATH=.:experiments/v4 python3 experiments/v6/loso_record_tests.py
run loso_transformer_tests env PYTHONPATH=.:experiments/v4 python3 experiments/v6/loso_transformer_tests.py

run confirmation bash run_confirm.sh
run tables python3 experiments/make_tables.py --tables tables --out tex_tables
run figures python3 experiments/make_figures.py --tables tables --out figures
echo "=== done === $(date +%H:%M:%S)"
