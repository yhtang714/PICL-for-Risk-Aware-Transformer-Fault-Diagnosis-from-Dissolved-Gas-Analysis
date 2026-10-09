#!/usr/bin/env bash
# Confirmation splits 62-71.
set -o pipefail
export OMP_NUM_THREADS=1 PYTHONPATH=.:experiments/v4
S="62 63 64 65 66 67 68 69 70 71"
S5="62 63 64 65 66"
run() { echo "=== $1 === $(date +%H:%M:%S)"; shift; "$@" 2>&1 | grep -v Warning | grep -v "warnings.warn"; }
mkdir -p tables/confirm

run run_seeds python3 experiments/run_seeds.py --seeds $S --out tables/confirm
run dump_scores python3 experiments/dump_scores.py --seeds $S
run main_table python3 experiments/main_table.py --seeds $S --out tables/confirm
run paired python3 experiments/paired_differences.py --seeds $S --out tables/confirm
run matched python3 experiments/matched_coverage.py --seeds $S --out tables/confirm
run missingness python3 experiments/missingness_patterns.py --seeds $S --out tables/confirm
run extract python3 experiments/v4/extract.py --seeds $S --out results/v4_confirm
run extract_loso python3 experiments/v4/extract.py --loso --seeds $S5 --out results/v4_confirm
run hierarchical python3 experiments/v4/eval_indist.py --seeds $S --out tables/confirm
run unseen_source python3 experiments/v4/eval_loso.py --seeds $S5 --out tables/confirm --root results/v4_confirm/loso
run recalibration python3 experiments/v5/eval_fewshot.py --seeds $S5 --out tables/confirm --root results/v4_confirm/loso
run catboost python3 experiments/v6/catboost_eval.py --seeds $S --root results/v4_confirm/indist --out tables/confirm
run record_tests python3 experiments/v6/indist_record_tests.py --seeds $S --tag _confirm
run record_tests_pooled python3 experiments/v6/indist_record_tests.py --seeds 52 53 54 55 56 57 58 59 60 61 $S --tag _pooled20
run transformer_tests_pooled env PYTHONPATH=.:experiments/v4 python3 experiments/v6/transformer_tests.py --tag _pooled20
