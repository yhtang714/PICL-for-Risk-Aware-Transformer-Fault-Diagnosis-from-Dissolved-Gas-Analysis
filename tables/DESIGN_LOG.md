# Design log

All variants below were compared on the design splits 42-51 (leave-one-source-out: design seeds 42-46).

The scripts for these exploratory studies are not part of this release. Their result files are kept as sheets of `design_studies.xlsx`; the `index` sheet gives the file name each sheet corresponds to.

## Decision layer and unseen-source read-out (experiments/v4)

Evidence files: tables/design/v4_*.csv (scripts in experiments/v4/).

### Not adopted (no consistent gain on the design splits)
- Random-intensity / scale-free evidence in the in-distribution read-out: accuracy +0.015 but AURC +0.029 (worse on 8 of 10 splits) -> v4_grid.csv, v4_w12_rivar.csv
- Head on log-ratios + evidence only (levels only through the SCM), in distribution: +0.008 accuracy, +0.003 AURC (neutral) -> v4_grid.csv
- Fusion / stacking on out-of-bag training predictions; learned selectors (OOB or calibration); Gini confidence -> v4_w1_stack.csv
- Hierarchical (family -> ordinal sub-class) head, with or without IEC monotone constraints -> v4_w1_hier.csv
- Extra-trees head -> v4_w1_explore.csv
- Source-specific intercepts in the SCM evidence: +0.010 accuracy, but baselines given the source indicator gain more (XGBoost 0.727) -> labelling conventions differ by source; not used (v4_w1_source2.csv)
- Two-fault likelihood-ratio gate from the SCM alone (moment-matched ppm-additive hypotheses): weak (misleading 0.224 -> 0.199 at 13 % flags) -> v4_w3_mixtures.csv
- Source-level cross-fitting of the fusion for unseen sources: higher accuracy (0.483) but worse AURC (0.454) -> v4_w2_final_design.csv

### Adopted
1. Hierarchical decision rule (all methods, same rule): sub-class reported for the 70 % most confident calibration records,
   mechanism family (PD | discharge D1-D2 | thermal T1-T3) reported for further records up to 95 % total coverage of the
   calibration records, the rest referred.  Design (PICL): decided 0.952 at 0.810 accuracy, errors 0.181 of all records
   (flat 85 % rule: 0.846 at 0.737, errors 0.223); two-fault mixtures misleading 0.170 (flat 0.224) -> v4_w3_hier.csv
2. Two-fault screen: random forest (500 trees, class-balanced, min leaf 2) on the read-out inputs (gases, log-ratios, evidence),
   real training records vs ppm-additive mixtures of complete training records of two different classes (lambda 0.1..0.9 step 0.1,
   15 pairs per class pair and lambda); flag threshold = 90th percentile of the calibration records' scores.
   Design: misleading 0.224 -> 0.168 (flat), 0.170 -> 0.133 (hierarchical) -> v4_w3_detector.csv, v4_w3_hier.csv
3. Unseen-source read-out (leave-one-source-out only): scale-free evidence (random fault intensity + common concentration factor,
   tau fitted by ML on complete training records), random forest on log-ratios + scale-free evidence, class- and source-balanced
   weights, fusion T and beta on the calibration split.  Design LOSO (micro, 5 seeds): accuracy 0.432 -> 0.472, macro-F1
   0.361 -> 0.410, AURC 0.434 -> 0.419, family accuracy 0.838 -> 0.852, family AURC 0.053 -> 0.049 -> v4_w2_final_design.csv

### Evaluation protocol (fixed)
- In distribution: splits 52-61, trained models unchanged (flat results identical to the base read-out); mixtures as in experiments/mixed_fault_ppm.py
  (lambda 0.25/0.5/0.75, 40 pairs per class pair, rng seed = split seed); baselines RF, XGBoost, SVM with the same rule.
- Screen generalisation: leave-one-class-pair-out training of the screen, flag rate on the held-out pair's mixtures.
- LOSO: seeds 52-56, same folds as experiments/loso.py; methods: PICL in-distribution read-out, PICL unseen-source read-out, RF, XGBoost, SVM.

## Local recalibration and further read-out variants (experiments/v5)

### Not adopted
- RF head hyper-parameters (max_features sqrt/0.5/1.0, min_samples_leaf 1/2/4): accuracy up to +0.013 but AURC never better than +0.002 (v5_w1_round2.csv)
- Heterogeneous calibrated ensemble (RF + SVM + XGB) as the head: with evidence acc 0.676 / AURC 0.167, without evidence 0.678 / 0.180 -> the ensemble, not the evidence, carries the accuracy; not adopted
- CO / CO2 as read-out inputs where measured (83 % of records): no gain (acc 0.655; on the CO subset 0.680 vs 0.700 without) (v5_w1_round2.csv)
- Evidence averaged over 5 SCM fits of the same split (different initialisations): the fits agree (mean |delta log q| 0.15 nats); AURC 0.174-0.175 vs 0.169 (v5_w1_bagged.csv)
- Two-fault identification head (multi-label RF trained on ppm-additive mixtures of training records, with or without two-fault SCM hypotheses as features): identifies the exact pair for 5-9 % of test mixtures, 0 % for a withheld class pair; real-record error rate 0.30 vs 0.175 for the hierarchical rule + screen (v5_w3_multilabel.csv)

### Adopted
4. Local recalibration of a transferred model with k labelled records of the new source: per-class logit bias b and temperature T fitted by
   ridge-penalised multiclass NLL (lambda = 1 / n) on the k records, softmax(log p / T + b); evaluated on the remaining records, R = 10 draws
   stratified by class, the same draws for every method.  Design LOSO (micro, seeds 42-46): PICL unseen-source read-out 0.472 -> 0.508 (k=10),
   0.532 (k=20), 0.543 (k=40); most accurate of all methods at every k; family accuracy 0.852 -> 0.897 (k=40); sub-class AURC stays slightly
   above SVM's (fewshot_design/v5_fewshot_summary.csv)

## Evidence combination, head learners and the decomposition-chain prior (experiments/v7)

### Not adopted (v7_arrays_per_seed.csv; read-out refinements with the model unchanged)
- Record-dependent evidence weight beta(x) gated on the evidence margin and/or the head's confidence: accuracy -0.007..+0.001, AURC +0.001..+0.006
- Per-class bias on the fused logits (vector scaling, ridge): accuracy -0.013, AURC +0.009
- Scale-free evidence as a second head input (with or without a second fusion weight): accuracy -0.003/-0.005, AURC +0.001/+0.007
- Agreement-based deferral (demote records whose fused family differs from the evidence family): AURC +0.000/+0.003
- IEC dominance-ordering penalty on the interventional response (physics_ordering_weight = 1): accuracy +0.002, AURC +0.003 (v7b/v7_retrain_per_seed.csv)

### Adopted on the design splits, reversed after one evaluation
5. Hydrocarbon decomposition chain as physics-plausible gas->gas edges (CH4->C2H6, C2H6->C2H4, C2H4->C2H2, pi = 0.55, reverse
   edges forbidden; config/prior_knowledge_chain.yaml).  Design splits (v7b/v7_retrain_per_seed.csv vs the base arm retrained in
   v7_retrain_per_seed.csv): accuracy 0.670 -> 0.675 (7/10), AURC 0.170 -> 0.162 (9/10), fitted evidence weight 0.10 -> 0.38; design
   LOSO (v7_loso_chain/): accuracy 0.472 -> 0.483, AURC 0.419 -> 0.387.  Because both design criteria were met it was adopted and the
   evaluation pipeline was started with it; on the evaluation splits 52-61 (v7_chain_eval_*.csv) accuracy fell to 0.656 (from 0.673)
   and AURC rose to 0.193 (from 0.188), below the identical-input random forest (0.658 / 0.202), so the change was reversed
   (config/prior_knowledge.yaml) and the evaluation run was stopped after the main table.
   The final configuration is unchanged and had already been scored on the confirmation splits 62-71, which were not
   touched by this variant.  Lesson recorded: a design-split gain of this size (p = 0.66-0.88 under the corrected test) is within the
   split-to-split noise of the SCM fit (the same prior retrained on the same split varies by up to 0.03 in accuracy), and the
   design criterion ("consistent direction on both metrics") is not sufficient at this effect size.

### Not adopted (v7_heads_per_seed.csv)
- Alternative head learners on the same 21 inputs with the same fusion: SVM head accuracy -0.011 / AURC +0.026; logistic
  regression -0.068 / +0.224; averages of forest and SVM (and logistic) heads within 0.002 of the forest in accuracy with AURC
  +0.002 / +0.012.
