# Numerical evidence for the threshold selection manuscript

This frozen release, `20261015_supervisor_r2`, supports Article 2, *Threshold Selection as a Critical Design Choice in Financial Fraud Detection*. It contains the selected evidence behind the reported ranking, threshold and review-workload comparisons, not the entire dissertation archive. The supplementary revision adds diagnostics without retraining models or replacing the primary fitted scores.

## Evidence scope

- `metrics/`: all **72 primary settings**, comprising five supervised models × seven strategies × two datasets and two separate OCSVM references; selected configurations, full-precision metrics and validation summaries accompany the grid.
- `thresholds/`: **12 baseline studies** using the four predefined rules on unchanged TEST scores. Classical thresholds aggregate five saved validation folds; FT-Transformer uses its selected internal validation checkpoint. The `supervisor_*` summaries add achieved precision, fold-cut sensitivity, the frozen FT A/B comparison and prior-adjustment diagnostics.
- `controls/`: predefined BAF absence-code and categorical-aware sampling controls, including paired comparison summaries, plus the `supervisor_*` supplementary reports. These do not select a replacement primary policy using TEST.
- `provenance/`: source selection, dataset and protocol identities, historical/current cost qualifications, sampler identity, LGBM score-resolution audits, metric policies and the article release inventory.
- `figures/` and `tables/`: the manuscript's selected graphical and numerical assets.
- `frozen/`: compressed saved score evidence for numerical recomputation, including baseline validation evidence and the score/label pairs required by the distributed analyses.

The release inventory specifies individual paths, hashes and sources. Archived run identifiers describe provenance; they are not a promise that the full original run directory is included. BAF Variant-transfer, SHAP and attention outputs belong to the companion study and are excluded here.

## Supplementary diagnostics

Seven named strategies remain visible across all 72 primary settings. SMOTE+Tomek and SMOTE are the same effective treatment in the audited executions because Tomek removes no rows. Aggregates use six effective conditions, counting that treatment once; this does not imply that the two algorithms are generally interchangeable.

The FT-Transformer diagnostic keeps the full-DEV model B as the primary result. The frozen validation pipeline A yields 3.903% alerts on VAL and 3.855% (7,710) on TEST, versus B's 5.707% (11,414) on the same TEST cases, at the cut `0.05843411013484001`. No TEST-based checkpoint choice is made. A/VAL versus B/TEST changes both observations and fitted pipeline; A/TEST versus B/TEST does not isolate the cause of their different operating points. A/TEST is distributed as an aggregate diagnostic, not a replayable checkpoint-and-score bundle.

The precision-target rule is a DEV selection objective, not a constraint guaranteed on TEST. FT-Transformer's achieved BAF TEST precision is 0.391, below the target of 0.5. The classical median fold threshold is an aggregation rule; fold spread and the saved cut-sensitivity table describe instability, not a confidence interval or prospective validation.

Average precision groups observations sharing the same score. Top-k precision and recall use expected uniform selection inside a boundary tie, with policy `expected_uniform_boundary_v1`, rather than a row-order-dependent sort. These checks do not infer a universal saturation cause.

The Brier diagnostic applies 15 analytical prior transformations with parameters derived from DEV. It is a supplementary sensitivity analysis, not fitted probability calibration or a new primary representation. A strictly increasing prior transform preserves the exact ranking mathematically; floating-point coalescence can create tiny AP or ROC-AUC differences, which the diagnostic records. Lower Brier values here do not establish calibrated probabilities or explain all effects of rebalancing.

The paired BAF CatBoost-minus-LightGBM AP difference is +0.0028, with a 1,000-resample paired TEST-row percentile interval of [-0.0017, +0.0072]. It is conditional on the fitted models and split, omits refitting and entity-cluster variability, and does not establish equivalence.

## Three different reproduction tasks

**Inspection and integrity verification** require neither raw data nor training. Run from the repository root:

```powershell
python src/article_release.py --verify
```

**Numerical recomputation** uses the distributed saved labels, scores and validation evidence rather than rerunning model fitting:

```powershell
python src/article_release.py --recompute --output-dir runs/recomputed
```

Use a new output directory and retain the frozen evidence. Recalculation can verify the numerical relationships supported by the distributed arrays; it cannot recreate measured historical wall-clock costs. The command replays the selected primary settings, controls and baseline threshold studies, not every supplementary diagnostic. The released diagnostic JSON/CSV/TeX summaries can be inspected and hash-verified, but recreating the A/TEST inference requires the omitted validation checkpoint and fitted preprocessors. Saved bootstrap reports retain their conditional TEST-row interpretation, not training-seed variability or established model equivalence.

Regenerate the three manuscript figures from the complete frozen scores and
threshold studies:

```powershell
python src/article2_figures.py --output-dir runs/article2_figures
```

This uses full precision–recall coordinates, not the compact curve samples.
PDF/PNG bytes can vary with rendering-library versions. No model is fitted.

**Independent training** additionally requires the provider datasets and scientific dependencies. Use the root README's explicit fresh output root and manifest. New searches and fits are not guaranteed to recover the manuscript's preserved historical BAF models or identical scores. Specialized correction/control queues still require their declared source artefacts.

## Boundaries and handling

Raw predictors and original CSVs are not distributed. Obtain ULB and BAF Base from their providers under the terms documented in `datasets/`. Fitted model weights, complete preprocessing artefacts, operational logs, checkpoints and the full author audit archive are omitted. Consequently this release supports numerical recomputation from selected predictions, not exact re-execution of every historical fitted function or all correction queues.

Historical CatBoost SMOTE records lack an explicit TEST-index artefact. Where paired evidence uses verified raw-file identity, the documented seed-42 split and matched ordered labels, the release preserves that qualification rather than inventing missing historical indices.

Average precision is computed from complete score evidence, not trapezoidal areas or compact plotting coordinates. Thresholds remain DEV-selected; recomputation must not optimize on TEST. The max-F2 threshold studies reproduce their pinned baseline confusion counts. Infeasible precision targets retain the recorded no-alert outcome. New top-k reports record their tie policy; historical score vectors are preserved rather than silently reordered or perturbed.

The inventory records an external reference to the [20-page manuscript revision reviewed on 15 October 2026](https://github.com/Olavo200100274/fraud-imbalance-classical-vs-transformers/blob/main/publications/article_2.pdf), with SHA-256 `a2dda3c4c7be17af5d0b331d5066d74598bbcc8008213c673994ee82f694d7f6`. The PDF is maintained in the main study repository and is not distributed in this checkout. Local inventory verification checks the released files and the reference metadata; it does not download or verify an external PDF. Any later PDF revision needs its own review and hash.

Do not train into or manually modify `results/`. A different release needs a new reviewed inventory. Integrity checks establish consistency of the selected evidence, not an independent audit of dataset authenticity, a fresh training replication or a production guarantee.
