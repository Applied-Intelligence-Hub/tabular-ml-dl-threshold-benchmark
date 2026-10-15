# Code organisation

The scientific code is retained separately from the private LaTeX projects.

| Area | Entry points / modules |
|---|---|
| Inputs and transformations | `data.py`, `preprocess.py`, `categorical_preprocess.py`, `missing_values.py` |
| Supervised and anomaly baselines | `main.py`, `main_transformer.py`, `models/`, `strategies/`, `evaluation/` |
| Selection and provenance | `experiment_protocol.py`, `ft_run_protocol.py`, `save_load.py` |
| Thresholds and transfer | `threshold_study.py`, `revision_thresholds.py`, `revision_transfer.py`, `cross_domain.py` |
| Compatible explanations | `shap_analysis.py`, `generate_revision_interpretability.py`, `revision_variant_shap.py`, `attention_analysis.py` |
| Prespecified controls and diagnostics | `revision_catboost_control.py`, `revision_paired_analysis.py`, `revision_missing_audit.py`, `revision_sampler_audit.py`, `revision_lgbm_diagnostics.py` |
| Supplementary diagnostics | `supervisor_threshold_diagnostics.py`, `supervisor_interpretability_diagnostics.py`, `supervisor_revision_audit.py` |
| Dedicated input/evidence checks | `verify_datasets.py`, `article_release.py` |
| Dedicated figure replay | `article2_figures.py`, `article2_study_overview.py`, `article2_evidence.py` |
| Wider-study reporting modules | `generate_results.py`, `export_public_evidence.py` |

`revision_*` and `run_revision_*` also preserve the provenance checks and
orchestration used for the October correction. These source-specific queues
require the complete local archive and explicit preserved-model pins; they
are not a one-command recipe for reproducing this release from the compact
public export. `revision_thesis_assets.py` is retained because reporting QA
uses its integrity checks; its optional LaTeX-patch mode additionally requires
the private writing project. No private writing sources are shipped here.

The `supervisor_*` scientific modules preserve the no-retraining supplementary
diagnostics and their source checks. They do not make the compact export a
complete author-side archive. Recreating the A/TEST inference in
`supervisor_threshold_diagnostics.py` requires the omitted frozen validation
checkpoint and its fitted preprocessors; its released aggregate report remains
inspectable. Interpretation/Variant modules are shared with the wider study,
but their outputs are outside this manuscript's evidence inventory. Editorial
handoff and Overleaf-packaging utilities are not scientific replay entry points
and are excluded from this public tree.

`evaluation/metrics.py` groups equal scores for average precision and uses
`expected_uniform_boundary_v1` for top-k metrics: a tie crossing the k boundary
contributes its expected fraud count under uniform selection within that tie.
This avoids row-order-dependent precision and recall without modifying the
saved scores. `generate_results.py` counts SMOTE/SMOTE+Tomek once when computing
aggregate strategy spread in these audited runs, while retaining both named
rows and all 72 primary settings.

The root README gives isolated fresh-training commands. Synthetic checks are
under `tests/` and `src/test_revision_*.py`; they do not retrain the dissertation
grid. Purely editorial PDF review utilities and superseded article plotting
scripts are excluded from the current public tree.

Use `article_release.py` for this repository's inventory and frozen-score
recomputation. `export_public_evidence.py` is retained as a shared dependency
and author-side exporter; its wider-thesis verification command expects a
different release manifest and is not the dedicated article checker.
