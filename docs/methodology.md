# Methodology

This document records the modelling methodology of the project. It is updated at each phase and is also part of the RAG assistant's knowledge base. **Numerical results are not copied here.** They are stored in `artifacts/metrics/*.json`, the source of truth for all reported figures.

Status: Phase 3 complete, including the one-time test evaluation. Sections marked *planned* are not yet implemented.

## 1. Validation protocol

| Partition | Size | Used for |
|---|---|---|
| Training | 18,000 (60%) | Fitting all learned transformations and models; 5-fold stratified cross-validation for hyperparameter tuning |
| Validation | 6,000 (20%) | Model comparison, decision-threshold selection, calibration-method selection |
| Test | 6,000 (20%) | Final, one-time evaluation of the selected models only |

- **Split:** stratified by the target, random seed 42. The client IDs of each partition are stored in `data/processed/splits.json`.
- **Leakage controls:**
  - Feature engineering, imputation, scaling and encoding all sit inside a single scikit-learn `Pipeline`. Cross-validation therefore refits them on each training fold.
  - The experiment loader (`credit_risk.experiment.load_partitions`) does not return test rows. A test enforces this.
- **Known residual risk:** 56 rows duplicate another row's features (often inactive accounts). Identical feature vectors can therefore appear in more than one partition. The effect on reported metrics is expected to be negligible (< 0.2% of rows).

## 2. Features

**Original variables.** All 23 explanatory variables are used, including `SEX`, `MARRIAGE` and `AGE`, at the project owner's request to maximise predictive performance (see the fairness diagnostics in Section 6). `ID` is excluded.

**Code merging (assumption).** Undocumented `EDUCATION` codes 0/5/6 are merged into 4 ("others"), and `MARRIAGE` 0 into 3 ("others").

**Engineered features** (`credit_risk.features.CreditFeatureEngineer`, row-wise and with no learned parameters):

| Feature | Definition |
|---|---|
| `utilization_1`, `utilization_mean`, `utilization_max` | `BILL_AMT / LIMIT_BAL` (latest month, 6-month mean, 6-month max), capped at ±10 |
| `bill_trend` | (`BILL_AMT1 − BILL_AMT6`) / `LIMIT_BAL` |
| `payment_to_bill_ratio` | Σ `PAY_AMT` / Σ max(`BILL_AMT`, 0), capped at 10 |
| `payment_1_to_limit` | `PAY_AMT1 / LIMIT_BAL` |
| `months_delayed`, `max_delay`, `recent_delays_3m` | Count, maximum and last-3-month count of months with `PAY_k ≥ 1` |
| `delay_trend` | max(`PAY_1`, 0) − max(`PAY_6`, 0) |
| `months_paid_duly` | Count of `PAY_k = −1` |
| `months_negative_bill`, `months_zero_bill`, `inactive_account` | Balance-pattern indicators |

Ratios use six-month aggregates, because the dataset does not document which bill each monthly payment settles.

## 3. Preprocessing

- **Logistic Regression:**
  - Monetary amounts get a signed log transform, sign(x)·log(1+|x|), then standardisation.
  - Other numeric inputs are standardised.
  - `SEX`, `EDUCATION` and `MARRIAGE` are one-hot encoded.
  - Repayment-status codes `PAY_k` are one-hot encoded too, with levels under 30 training rows pooled. The default rate is not monotonic in the raw code, so one linear coefficient per code would be misspecified.
  - Median or most-frequent imputation is included so the model handles incomplete inputs. The dataset itself has no missing values.
- **Tree models** (Random Forest, XGBoost, imbalance-aware ensembles): raw numeric inputs with median imputation. Tree splits are invariant to monotonic transforms.

## 4. Models

| Name | Description |
|---|---|
| `dummy_prior` | Predicts the training default rate for every client (no-skill reference) |
| `logreg_raw` | L2-regularised Logistic Regression on the original variables |
| `logreg` | As above, plus the engineered features |
| `logreg_balanced` | As `logreg`, with `class_weight="balanced"`, used to study class reweighting |
| `random_forest` | Random forest, 300 trees; `max_depth`, `min_samples_leaf`, `max_features` by randomised search (12 candidates) |
| `xgboost` | XGBoost (`hist`, CPU); 8 hyperparameters by randomised search (30 candidates) |

Hyperparameters (regularisation strength `C`) are tuned by grid search with 5-fold stratified CV on the training data. The tuning score is **average precision**: threshold-free and focused on the minority class. Calibration is assessed separately, because average precision ignores it.

**Class imbalance.** About 22% of clients default. Standard imbalance strategies are evaluated, not assumed (`src/credit_risk/imbalance.py`, `configs/models.yaml`, stage 2). Each strategy variant **inherits the tuned hyperparameters of its base model** and changes only the strategy, so the comparison isolates the strategy's effect.

| Family | Methods evaluated | Notes |
|---|---|---|
| Cost-sensitive learning | `class_weight="balanced"` (LR), `balanced_subsample` (RF), `scale_pos_weight = N₋/N₊ ≈ 3.52` (XGBoost) | Weight w_c = N / (K·N_c) |
| Oversampling | Random oversampling, SMOTE-NC | SMOTE-NC treats `SEX`, `EDUCATION`, `MARRIAGE`, `PAY_1..6` as categorical |
| Undersampling / cleaning | Random undersampling, Tomek links, Edited Nearest Neighbours | |
| Hybrid | SMOTE-NC + Tomek, SMOTE-NC + ENN | |
| Imbalance-aware ensembles | Balanced Random Forest, EasyEnsemble, RUSBoost | |
| Decision threshold | F-beta threshold on validation (Section 6) | Changes no data and no model |

Implementation rules:
- **Pipeline placement.** Resampling runs inside an `imblearn.pipeline.Pipeline`. It is applied only in `fit`, i.e. to the training data and to the training folds during CV. Validation and test data keep their natural distribution.
- **Resample before feature engineering.** Synthetic clients are generated on the raw variables, and engineered features (delay counts, utilisation) are recomputed from each synthetic client. That keeps derived features consistent with the synthetic repayment history.
- **Scaling for neighbour search.** Nearest-neighbour methods search on signed-log, standardised numeric variables, using statistics from the fold being resampled. Results are mapped back to original units.
- **Excluded methods.** ADASYN, Borderline-SMOTE, SVM-SMOTE and KMeans-SMOTE interpolate every column, including the categorical codes, so they are excluded. NearMiss is excluded because of its known sensitivity to noise. Focal loss is excluded because it is mainly motivated by settings where easy examples dominate massively (e.g. object detection). For tabular credit data, class weights are the standard cost-sensitive baseline.

Resampling and class weighting change the class prior the model learns from. Their scores are then **not** PDs until recalibrated.

## 4b. Probability calibration

Calibration (`src/credit_risk/calibration.py`) uses `CalibratedClassifierCV(ensemble=False)` with 5-fold stratified CV **on the training set**:
1. Out-of-fold scores are produced for all training rows. Each fold model is the full pipeline, including any resampling.
2. A calibration map is fitted to those scores: **sigmoid** (Platt; 2 parameters) or **isotonic** (non-parametric, monotone; 18,000 out-of-fold scores support it).
3. The base pipeline is refitted on the full training set, and the map is applied to it.

Out-of-fold rows keep the natural default rate, so the map also corrects the prior shift from resampling or reweighting. The calibration method is chosen on validation (Brier score, log loss, ECE). An analytic alternative for pure random under-/oversampling is the prior correction odds_true = odds_model × [π/(1−π)] / [π_s/(1−π_s)], where π_s is the training-sample default rate after resampling. It is not used because it does not hold exactly for SMOTE, cleaning methods or ensembles.

## 4c. Model selection rule

Declared before the final Phase 3 run and implemented in `src/credit_risk/selection.py`, using validation results only:
1. **Eligible:** ECE ≤ 0.03 and |mean predicted PD − observed default rate| ≤ 0.02, i.e. the scores are usable as PDs.
2. **Primary criterion:** highest PR-AUC among eligible models.
3. **Parsimony:** among eligible models whose PR-AUC is not significantly below the best (paired bootstrap 95% interval includes 0), choose the simplest. The order is logistic regression < random forest < gradient boosting; within a family, no resampling < class weights < resampling, then no calibration < sigmoid < isotonic.

The best-PR-AUC model and the parsimonious choice are both reported. The output is `artifacts/metrics/selection.json`.

## 5. Metrics

| Metric | Measures | Strengths | Limitations |
|---|---|---|---|
| ROC-AUC | Probability that a random defaulter is scored above a random non-defaulter | Threshold-free; insensitive to prevalence | Ignores calibration; can look good while precision in the high-risk region is poor |
| PR-AUC (average precision) | Precision averaged over recall levels | Focuses on the minority class; sensitive to performance at the top of the ranking | Depends on prevalence (no-skill value = default rate); harder to compare across datasets |
| Brier score | Mean squared error of PDs, (1/n) Σ (p − y)² | Proper scoring rule; rewards both calibration and discrimination | Hard to interpret alone; compare with the base-rate reference π(1 − π) via the Brier skill score |
| Log loss | Mean negative log-likelihood | Proper scoring rule; strongly penalises confident errors | Unbounded; dominated by a few extreme errors |
| Expected calibration error (ECE) | Count-weighted mean gap between observed rate and mean PD over 10 equal-frequency bins | Direct summary of calibration | Depends on binning; not a proper scoring rule |
| Precision, recall, F1, F2 | Decision quality at one threshold | Directly interpretable for a decision rule | Depend on the threshold and the prevalence; say nothing about the PDs themselves |

Accuracy is not used for model selection: with a 22% default rate, predicting "no default" for everyone scores about 78%.

**Uncertainty.**
- 95% percentile bootstrap intervals (1,000 resamples of the validation set, model and threshold held fixed) are reported for the main metrics.
- Model differences are tested with a **paired** bootstrap on the same resamples (`artifacts/metrics/comparisons.json`). This removes shared sampling noise.
- These intervals don't capture variability from retraining the models.

## 6. Decision threshold

- **How it's chosen:** maximise **F2** on the validation set, over a fixed grid with step 0.005. β = 2 weights recall more than precision, as the project owner requested. Ties go to the lower threshold.
- **Capacity-constrained rule.** The same F2 objective restricted to thresholds that flag at most 30% of clients (`threshold.max_flagged_rate`). 30% is an illustrative operating assumption, not a business or regulatory figure. Both rules are stored for every model.
- **0.5 is not optimal.** It has no special status for a calibrated PD at a 22% default rate. Metrics at 0.5 are stored for reference only.
- **Caveat:** at this prevalence, flagging every client already gives F2 = 5π / (4π + 1) ≈ 0.59, since recall = 1. F2-optimal thresholds therefore tend to flag a large share of clients. Operational use would need a capacity constraint (maximum share flagged) or explicit, documented costs.
- **Costs:** `credit_risk.decision_analysis.expected_cost` uses hypothetical unit costs for false negatives and false positives. The dataset has no exposure or loss data, so any resulting cost figure is illustrative only.

## 7. Fairness diagnostics

For `SEX` and age bands, the project reports per group:
- the observed default rate and mean predicted PD (calibration-in-the-large);
- the share of clients flagged, recall, false-positive rate and ROC-AUC.

These are descriptive diagnostics, not a legal fair-lending assessment.

## 8. Interpretation rules

- Coefficients and (later) SHAP values describe **predictive associations** learned from historical data. They are not causal effects.
- Correlated inputs (monthly bills, delay codes and their counts) share predictive signal, so an individual coefficient's sign or size can be unstable.

## 8b. Phase 3 results (validation)

Figures are read from `artifacts/metrics/` (run of 2026-10-29 on the project owner's PC). See notebooks 04 and 05.

- **Model families.**
  - XGBoost outperforms Logistic Regression in training-set CV (PR-AUC 0.569 vs 0.548) and on validation (paired ΔPR-AUC +0.016 [+0.001, +0.030]).
  - XGBoost and Random Forest are equivalent in CV (0.569 vs 0.570). XGBoost leads slightly on validation (+0.006 [+0.001, +0.011]).
  - Paired intervals are not adjusted for the ~34 variants compared on validation. The selected model's validation score is therefore optimistically biased, which is why the frozen model is evaluated once on the test set (Section 8c).
- **Class-imbalance strategies.** On this dataset and protocol, none of the tested methods improved discrimination over training on the original distribution.
  - Class weights and Tomek links are indistinguishable from the base model.
  - SMOTE-NC, random undersampling, the hybrids and the imbalance-aware ensembles are lower.
  - This is an empirical finding for these models and data, not a general claim that a 22% default rate makes rebalancing useless.
- **Why rebalancing fails here.**
  - (i) *Prior shift.* Balanced training multiplies the model's odds by [π_s/(1−π_s)]/[π/(1−π)] ≈ 3.5. Dividing this factor out analytically (`calibration.prior_correction`, no fitted parameters) brings ECE from about 0.20 to 0.01–0.03, so the miscalibration is almost entirely explained by the shift.
  - (ii) *Synthetic interpolation.* SMOTE-NC loses ranking quality that random oversampling (same prior shift, exact copies) does not: ROC-AUC 0.776 vs 0.783. The extra loss is attributable to the synthetic points. The mechanism (interpolated defaulters landing in regions dominated by non-defaulters) is a hypothesis.
  - (iii) *Reweighting cannot change the ranking much.* Weighting and resampling change error costs, not ranking information. Moving the threshold of the unweighted model produces the same precision–recall trade-off.
- **Calibration.**
  - The unmodified XGBoost and Logistic Regression are already well calibrated (validation ECE ≈ 0.015). Post-hoc calibration does not improve their Brier score, and isotonic calibration lowers PR-AUC through ties.
  - Resampled models need recalibration. Notebook 05 reports raw, isotonic and analytic-correction results side by side, with the same metrics on the same validation set.
- **Selected model:** `xgboost`, with no resampling and no post-hoc calibration (rule of Section 4c; `selection.json`).

## 8c. Final test evaluation (one-time)

- **Protocol.** The model and both thresholds (F2 = 0.135; F2 with ≤ 30% flagged = 0.230; both chosen on validation) were frozen in `frozen_decision.json`, with model-file hashes, before the test set was read. `scripts/evaluate_test.py` ran once (`test_access_count = 1`). Reference models are reported for context only.
- **Results, XGBoost on 6,000 test clients:**
  - ROC-AUC 0.783 [0.770, 0.798], PR-AUC 0.564 [0.538, 0.595], Brier 0.1342, ECE 0.007. These are close to validation (0.788 / 0.563 / 0.1333).
  - The capacity rule flags 29.4%, with recall 0.625 and precision 0.470.
  - Paired against Logistic Regression: ΔPR-AUC +0.037 [+0.021, +0.052]. Against Random Forest: +0.007 [+0.001, +0.012].
  - In every PD band, the mean PD is within about 1.5 points of the observed rate. The top band (PD ≥ 40%) holds 15% of clients and 43% of defaults.
- **Error analysis.** Missed defaulters have no recorded delay and low utilisation. They are indistinguishable from non-defaulters on the available variables, which is a limit of the data rather than of the threshold.
- **Use of the test set.** It has now been used. Further model changes would need new, untouched data for an unbiased estimate.

## 8d. Explainability (SHAP)

- **Method.** Exact TreeSHAP (`shap.TreeExplainer`) on the frozen XGBoost model, computed for the 6,000 test clients (`scripts/compute_shap.py`). The model is unchanged and no decision is re-made.
- **Scale.** Values are in **log-odds**: base value + Σ SHAP = logit(PD). Additivity is checked (maximum gap 2.7e-6).
- **Global drivers.** By mean |Σ SHAP| within variable groups:
  - repayment-delay history (engineered counts 37%, raw `PAY_k` codes 19%);
  - utilisation and payment ratios (14%);
  - payment amounts (13%) and bill amounts (9%);
  - credit limit (6%);
  - demographics (2%).
- **Consistency check.** The ranking matches the univariate analysis (notebook 02) and the dominant Logistic Regression coefficients (repayment-status levels). Higher `LIMIT_BAL` and higher payments are associated with lower PD; delays and high utilisation with higher PD.
- **Interpretation rules.**
  - SHAP values are predictive associations learned from 2005 data, not causal effects.
  - Correlated features share credit, so grouped totals are reported alongside individual features.
  - Individual explanations in the dashboard are for test-set or hypothetical clients. They are not assessments of real people.

## 9. Planned (later phases)

RAG assistant (Phase 6); final report.

## 10. Compute

Experiments run on the project owner's laptop: 20 logical CPU threads, Windows 11. XGBoost uses the CPU `hist` method. The GPU (6 GB) is reserved for the local LLM. On 18,000 training rows, the CPU with parallel CV is faster than a single GPU process. `device: cuda` can be set in the XGBoost parameters for comparison.
