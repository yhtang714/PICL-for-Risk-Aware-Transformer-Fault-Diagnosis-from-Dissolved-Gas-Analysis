# PICL

Code and data for the paper "Physics-Informed Causal Learning for Risk-Aware Transformer Fault Diagnosis from Dissolved Gas Analysis" (Results in Engineering, RINENG-D-26-17624).

The method fits a linear-Gaussian causal model of how faults produce gases. Relations documented in IEC 60599 are fixed, uncertain ones get prior probabilities, and the rest is learned from six data sources. For each candidate fault, the model gives the likelihood of the measured gases when only that fault is active. This evidence is passed to a random forest, the two are combined on a calibration split, and each record is then given a sub-class, a fault family, or sent to an expert.

## Setup

Python 3.10 or newer.

    pip install -r requirements.txt

Run all commands from this folder with `PYTHONPATH=.` and `OMP_NUM_THREADS=1`.

## Data

`data/dga_provenance.csv` contains every record used in the paper, one row per DGA sample. Gas concentrations (H2, CH4, C2H2, C2H4, C2H6, CO, CO2) are in uL/L; an empty cell means the gas was not measured. Other columns give the source, the transformer or case ID, the fault label, how the label was obtained, and whether the record is a cross-source duplicate.

The 638 records used for training and testing are selected by `data.filter_query` in `config/config.yaml`. The 412 rule-labelled Fujian records and the 71 records without a diagnosed fault are only used by `weak_label_control.py` and `nonfault_screening.py`.

## Quick start

Train and test on one split (under a minute on two CPU cores):

    PYTHONPATH=. python train.py --seed 52

Rebuild the LaTeX tables and the figures from the stored results (no training needed). The tables are written to `tex_tables/`:

    PYTHONPATH=. python experiments/make_tables.py --tables tables --out tex_tables
    PYTHONPATH=. python experiments/make_figures.py --tables tables --out figures

## Full reproduction

    bash run_all.sh

This retrains everything and rewrites the results, tables and figures. It takes several hours on two cores.

Each split is set by a seed, which fixes both the data partition and the model initialisation. Seeds 42-51 are design splits, 52-61 evaluation splits and 62-71 confirmation splits (`run_confirm.sh`). Leave-one-source-out runs use seeds 52-56.

Scripts behind each table and figure of the main text:

| Paper | Scripts (in `experiments/`) |
|---|---|
| Table 3 | `config/prior_knowledge.yaml` |
| Table 5 | `make_tables.py` |
| Table 7 | `config/config.yaml` (PICL); baseline settings are in the scripts that train them |
| Table 8 | `kl_sensitivity.py` |
| Table 9, Figure 5 | `knowledge_alignment.py`, `graph_transition.py` |
| Table 10, Figure 6 | `main_table.py`, `metrics_full.py`, `modern_baselines.py`, `v6/catboost_eval.py` |
| Table 11 | `paired_differences.py`, `v6/indist_record_tests.py`, `v6/transformer_tests.py` |
| Table 12 | `weight_sensitivity.py`, `transform_alternatives.py` |
| Table 13, Figure 7 | `dump_scores.py`, `matched_coverage.py` |
| Figure 8 | `misspecification_test.py` |
| Tables 14-15 | `residual_diagnostics.py`, `scm_variants.py`, `picl_v3_explore.py` |
| Table 16, Figure 9 | `counterfactual_faithfulness.py`, `intervention_magnitude.py` |
| Tables 17-19, Figure 10 (top) | `loso.py`, `v4/extract.py --loso`, `v4/eval_loso.py`, `v5/eval_fewshot.py`, `v6/loso_transformer_tests.py` |
| Table 20 | `weak_label_control.py` |
| Table 21, Figure 10 (bottom) | `missingness_patterns.py` |
| Table 22 | `v4/extract.py`, `v4/eval_indist.py` |
| Table 23 | `matched_coverage.py`, `threshold_analysis.py`, `complexity_scalability.py` |
| Table 24 | `run_confirm.sh`, `v6/transformer_tests.py --tag _pooled20` |
| Tables 25 and 27 | `ablation_matched.py`, `causal_necessity.py` |
| Table 26 | `learning_curve.py` |

Supplementary Material: Table S2 comes from `picl_v2_explore.py`, Figure S2 from `weight_sensitivity.py`, Table S3 from `nonfault_screening.py`, and Table S4 and Figure S3 from `metrics_full.py`. The variants listed in Section S6 are summarised in `tables/design/` (see below).

## Stored results

- `tables/` holds the CSV files that the table and figure scripts read.
- Results that no script reads back are collected in `tables/supporting_results.xlsx` and, for the confirmation splits, `tables/confirm/confirm_results.xlsx`. Each sheet is one result file; the `index` sheet gives its original file name. Rerunning a script writes that file as a CSV again.
- `results/scores/` has the per-record scores of every split (one file each for calibration and test records, plus `meta.json`).
- `results/arrays.zip` holds the arrays exported by `v4/extract.py`. The scripts read them directly from the archive; files written by a new run of `extract.py` are used instead.

## Folders

    picl/          the method (data, graph, SCM, learning, evidence, classifier, calibration, decisions)
    experiments/   one script per experiment; shared helpers in _common.py
      v4/          array export, hierarchical decisions, two-fault screen, unseen-source read-out
      v5/          local recalibration on a new source
      v6/          CatBoost baseline and record-level tests
    config/        settings and the IEC 60599 prior knowledge
    data/          the DGA records
    tables/        stored results
    results/       per-record scores and exported arrays
    figures/       figures of the paper (PDF)

## Design studies

The classifier and the decision rules were chosen on the design splits 42-51. `tables/design/DESIGN_LOG.md` lists what was tried and what was kept. The read-out study behind Table S2 can be rerun with:

    PYTHONPATH=. python experiments/run_seeds.py --seeds 42 43 44 45 46 47 48 49 50 51 --out tables/design
    PYTHONPATH=. python experiments/picl_v2_explore.py --seeds 42 43 44 45 46 47 48 49 50 51 --out tables/design
    PYTHONPATH=. python experiments/picl_v3_reduced.py --seeds 42 43 44 45 46 47 48 49 50 51

The scripts of the other exploratory studies are not included; their results are in `tables/design/design_studies.xlsx`.

## Notes

Trained models are not included. `run_seeds.py` saves them to `results/seeds/` (about 60 MB per split).

Results are deterministic for a given seed and library versions. The stored results were produced with Python 3.11, NumPy 2.4, scikit-learn 1.8, PyTorch 2.14, XGBoost 3.2 and CatBoost 1.2.
