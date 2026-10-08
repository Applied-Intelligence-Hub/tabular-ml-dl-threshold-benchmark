# artigo 2

Code and evidence for *Threshold Selection as a Critical Design Choice in Financial Fraud Detection: A Systematic Comparison of Classical Machine Learning Models and Tabular Transformers*.

The repository separates three decisions that fraud-detection benchmarks often blur together: how a model is fitted, how its operating threshold is chosen, and how the resulting detector is evaluated. All model and threshold selection happens on development data only; test scores are fixed before any threshold rule is applied.

## What this compares

Five supervised model families plus a one-class reference, evaluated on two datasets with different feature representations and fraud prevalence:

- **Logistic Regression, Random Forest, LightGBM, CatBoost** — classical baselines, tuned with Optuna (50-trial budget, five-fold validation)
- **FT-Transformer** — tabular transformer baseline, tuned with an internal holdout
- **One-Class SVM** — unsupervised reference, fitted on legitimate cases only

Datasets:

- **ULB 2013** (credit card transactions, PCA features, 283,726 distinct profiles after exact-duplicate removal)
- **BAF Base** (synthetic bank account applications, semantic numerical and categorical features, 1,000,000 rows)

## Two separate experiments

1. **Imbalance study** — seven strategies (None, RUS, ROS, SMOTE, SMOTE+Tomek, SMOTEENN, Class Weights) across all five supervised models and both datasets, reusing baseline-selected hyperparameters.
2. **Threshold study** — four pre-specified rules (fixed 0.5, validation max-F1, validation max-F2, validation precision >= 0.5) applied to each baseline's saved, unchanged test scores. This isolates the effect of the decision cut from the effect of the fitted model.

A predefined set of representation controls (absence-code handling, SMOTE-NC for categorical features) is reported separately from the primary factorial and does not feed back into the primary configuration choices.

## Key results

- On BAF, the fixed 0.5 threshold gives F2 scores of roughly 0.001-0.048 across supervised models; validation-selected max-F2 raises this to 0.287-0.316, at the cost of a substantially larger alert volume (recall and alert rate move together).
- No imbalance intervention consistently improves ranking (PR-AUC) across models and datasets; SMOTE reduces BAF FT-Transformer PR-AUC by roughly 35%, with preprocessing differences preventing a clean architecture-only explanation.
- Top-tier BAF baselines (CatBoost 0.1796, FT-Transformer 0.1769, LightGBM 0.1768 PR-AUC) are close enough that bootstrap intervals overlap broadly; point-estimate rankings should not be read as established superiority.

Full tables, figures and the configuration-guidance summary are in the manuscript.

## Repository structure

```
.
├── data/               # dataset loaders and the exact-profile deduplication / split logic
├── preprocessing/       # imputation, encoding, resampling pipelines (per model family)
├── models/               # LR, RF, LightGBM, CatBoost, FT-Transformer, OCSVM
├── search/               # Optuna search spaces and tuning scripts
├── evaluation/           # threshold rules, metrics, bootstrap intervals
├── figures/               # plotting scripts (PR curves, imbalance heatmaps, study overview)
├── results/               # saved per-run metrics, configs, and score arrays
└── results_revision/     # dated manifest and reproducibility audit trail
```

Each run's `config.json` records its hyperparameters and split provenance. The dated manifest under `results_revision/` pins the configurations, scores and figure sources used in the article.

## Reproducing the results

```bash
git clone https://github.com/<org>/tabular-ml-dl-threshold-benchmark.git
cd tabular-ml-dl-threshold-benchmark
pip install -r requirements.txt
```

Dataset access:

- ULB credit card data: [Kaggle, Machine Learning Group ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- BAF dataset suite: [Feedzai Research](https://github.com/feedzai/bank-account-fraud)

Both are subject to their respective providers' terms and are not redistributed here.

```bash
python -m data.prepare --dataset ulb
python -m data.prepare --dataset baf

python -m search.run --model lightgbm --dataset baf
python -m evaluation.threshold_study --model lightgbm --dataset baf
python -m figures.study_overview
```

Exact search spaces, environment details and per-script arguments are documented inline in each module.

## Scope and limitations

- Thresholds, hyperparameters and resampling are selected on development data only; nothing is selected on the held-out test set.
- BAF months are pooled rather than split chronologically — this is not a prospective temporal evaluation.
- Hyperparameters are selected once under the unmodified baseline and reused across imbalance strategies; strategies are not independently retuned.
- Reported bootstrap intervals describe test-row uncertainty conditional on the fitted model and split; they do not capture training-seed or re-split variability.
- F2 is a recall-weighted criterion, not a monetary cost function, and alert rate is a workload proxy, not a measurement of analyst time or prevented loss.

See the manuscript's Discussion and Limitations sections for the full account.

## Citation

If you use this code or the reported evidence, please cite the manuscript (citation details to be added on publication).

## Generative AI disclosure

OpenAI Codex assisted with manuscript revision checks against saved experimental evidence, and with reproducible plotting and layout code. Experimental values are read from recorded evaluations, not generated or inferred by a language model. The authors are responsible for the accuracy and interpretation of all reported results.

## License

Add a license (e.g. MIT, Apache-2.0) before making the repository public.
