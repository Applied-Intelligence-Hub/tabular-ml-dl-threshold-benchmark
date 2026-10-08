# Numerical evidence for the threshold selection manuscript

This frozen release supports Article 2, *Threshold Selection as a Critical Design Choice in Financial Fraud Detection*. It contains the selected evidence behind the reported ranking, threshold and review-workload comparisons, not the entire dissertation archive.

## Evidence scope

- `metrics/`: all **72 primary settings**, comprising five supervised models × seven strategies × two datasets and two separate OCSVM references; selected configurations, full-precision metrics and validation summaries accompany the grid.
- `thresholds/`: **12 baseline studies** using the four predefined rules on unchanged TEST scores. Classical thresholds aggregate five saved validation folds; FT-Transformer uses its selected internal validation checkpoint.
- `controls/`: predefined BAF absence-code and categorical-aware sampling controls, including paired comparison summaries. These do not select a replacement primary policy using TEST.
- `provenance/`: source selection, dataset and protocol identities, historical/current cost qualifications, sampler identity, LGBM score-resolution audits and the article release inventory.
- `figures/` and `tables/`: the manuscript's selected graphical and numerical assets.
- `frozen/`: compressed saved score evidence for numerical recomputation, including baseline validation evidence and the score/label pairs required by the distributed analyses.

The release inventory specifies individual paths, hashes and sources. Archived run identifiers describe provenance; they are not a promise that the full original run directory is included. BAF Variant-transfer, SHAP and attention outputs belong to the companion study and are excluded here.

## Three different reproduction tasks

**Inspection and integrity verification** require neither raw data nor training. Run from the repository root:

```powershell
python src/article_release.py --verify
```

**Numerical recomputation** uses the distributed saved labels, scores and validation evidence rather than rerunning model fitting:

```powershell
python src/article_release.py --recompute --output-dir runs/recomputed
```

Use a new output directory and retain the frozen evidence. Recalculation can verify the numerical relationships supported by the distributed arrays; it cannot recreate measured historical wall-clock costs. Saved bootstrap reports retain their conditional TEST-row interpretation, not training-seed variability or established model equivalence.

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

Average precision is computed from complete score evidence, not trapezoidal areas or compact plotting coordinates. Thresholds remain DEV-selected; recomputation must not optimize on TEST. The max-F2 threshold studies reproduce their pinned baseline confusion counts. Infeasible precision targets retain the recorded no-alert outcome.

The PDF hash in the inventory identifies the included author-review export. Repository-link revisions do not change the underlying experiments; a new PDF must be reviewed and repinned before it replaces the current export.

Do not train into or manually modify `results/`. A different release needs a new reviewed inventory. Integrity checks establish consistency of the selected evidence, not an independent audit of dataset authenticity, a fresh training replication or a production guarantee.
