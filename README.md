# Threshold Selection as a Critical Design Choice in Financial Fraud Detection

*A Systematic Comparison of Classical Machine Learning Models and Tabular Transformers*

Research by **Olavo Miguel Cabaço Caixeiro, Maryam Abbasi and Pedro Miguel de Oliveira Martins**, Universidade Politécnica de Santarém.

[Read the manuscript](publications/article_2.pdf) · [Inspect the evidence](results/) · [Explore the code](src/) · [Obtain the datasets](datasets/)

This dedicated repository accompanies **Article 2**, a manuscript in preparation for submission, not an accepted or published journal article. It separates model ranking, validation-selected operating thresholds and the resulting alert workload. The reviewed 8 October 2026 PDF includes the dedicated-repository links and identifies the public figure scripts and frozen inputs. These documentary updates do not change the reported experiments.

## Study and findings

Five supervised families (Logistic Regression, Random Forest, LightGBM, CatBoost and FT-Transformer) and a One-Class SVM reference are evaluated on ULB credit-card transactions and BAF Base bank-account applications. OCSVM fits legitimate examples only, but labelled validation examples select its threshold; it is not an entirely label-free workflow.

The **imbalance factorial** covers seven strategies for the five supervised models on two datasets, plus two separate OCSVM baselines: **72 primary settings**. The **threshold study** applies four predefined rules to the same saved baseline scores: fixed 0.5, validation max-F1, validation max-F2 and the lowest feasible validation threshold with precision at least 0.5. Its **12 baseline studies do not refit models** or select thresholds on TEST. Representation controls remain separate from the primary factorial.

| BAF FT-Transformer operating rule | F2 | Recall | Alerts among 200,000 applications |
|---|---:|---:|---:|
| Fixed threshold 0.5 | 0.048 | 3.94% | 149 |
| DEV-selected max-F2 | 0.304 | 55.80% | 11,414 |

The approximately 6.3-fold F2 increase changes the decision cut, not the ranking, and requires substantially more review. Across supervised BAF baselines, fixed-threshold F2 is 0.001–0.048, compared with 0.287–0.316 under max-F2.

- LightGBM has the highest ULB baseline average precision, 0.8205. BAF baseline point estimates are closely grouped for CatBoost (0.1796), FT-Transformer (0.1769) and LightGBM (0.1768); overlapping marginal intervals do not establish equivalence.
- No imbalance intervention improves ranking consistently across models and datasets. BAF FT-Transformer AP falls from 0.1769 to 0.1149 under primary SMOTE; the SMOTE-NC control reaches 0.1157, not baseline performance. Different sampling representations prevent an architecture-only explanation.
- A validation precision target does not guarantee TEST precision or a fixed alert budget. BAF FT-Transformer achieves TEST precision 0.391 under that rule.

PR-AUC denotes **average precision**, not trapezoidal integration. Candidate configurations in the manuscript are conditional summaries of jointly evaluated settings, not prospectively validated deployment prescriptions.

## Experimental protocol

ULB predictor-profile deduplication precedes the seed-42 stratified 80/20 DEV/TEST split, retaining 283,726 profiles and 473 frauds. ULB TEST contains 56,746 profiles and 95 frauds. BAF Base has one million applications; TEST contains 200,000 applications and 2,206 frauds. BAF months are pooled, not evaluated prospectively.

Supervised baseline selection uses 50 attempted Optuna trials: five-fold inner validation for classical models and an internal holdout for FT-Transformer. Corrected ULB searches are repeated after deduplication. Original BAF-selected parameters are retained, classical final models are preserved and FT-Transformer is refitted. Baseline-selected parameters are reused across imbalance interventions. Preprocessing and resampling use the corresponding training partitions only; thresholds are selected within DEV.

One fixed outer split and conditional TEST-row bootstraps do not measure training-seed or tuning variability. BAF is synthetic, CatBoost receives one-hot classifier inputs, and transformer SMOTE interpolates ordinal category codes. F2 is a recall-weighted criterion, not a monetary utility; alert volume is a workload proxy, not analyst time or prevented loss. Recorded CPU/GPU costs reflect heterogeneous workflows, not intrinsic architectural efficiency.

## Repository contents

```text
datasets/       Official input sources and exact hashes for ULB and BAF Base
notebooks/      ULB and BAF Base exploratory analyses
src/            Shared scientific code and dedicated article evidence checks
tests/          Synthetic protocol and evidence integrity checks
results/        Selected metrics, thresholds, controls, frozen scores and paper assets
publications/   This manuscript PDF
```

[results/README.md](results/README.md) describes the numerical evidence and its limits. Shared code is duplicated intentionally so this repository does not depend on another checkout. Some shared modules support the wider thesis workflow; Variant-transfer, SHAP and attention results are outside this paper's exported evidence.

## Setup and inputs

Use Python 3.12. The recorded correction environment was Python 3.12.14 on Windows 11, an Intel Core i5-13600KF, 16 GB RAM and an NVIDIA RTX 3070 with 8 GB VRAM. These are recorded conditions, not certified minimum requirements.

```powershell
git clone https://github.com/Applied-Intelligence-Hub/tabular-ml-dl-threshold-benchmark.git
cd tabular-ml-dl-threshold-benchmark
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python -m pip install -r requirements.txt
python -m pip check
```

The requirements include CUDA 12.1 PyTorch 2.5.1. For a CPU installation, retain the remaining pins and install the corresponding CPU PyTorch build instead of the CUDA wheel. Fresh installation on another computer has not been tested, and hardware changes need not yield identical trained models.

Raw CSVs are not redistributed. Follow [datasets/README.md](datasets/README.md) for provider downloads, filenames, terms and hashes. To check the two inputs used here:

```powershell
python src/verify_datasets.py --files creditcard_2013.csv Base.csv
```

## Inspect and recompute numerical results

Neither step below fits a model or requires the raw datasets. Integrity verification checks the pinned release inventory. Numerical recomputation uses the distributed frozen predictions and validation evidence rather than compact plotting curves.

```powershell
python src/article_release.py --verify
python src/article_release.py --recompute --output-dir runs/recomputed
```

Regenerate the three manuscript figures from the complete frozen scores and
threshold studies, without fitting a model:

```powershell
python src/article2_figures.py --output-dir runs/article2_figures
```

Rendering-library versions can change output bytes; numerical inputs remain
the pinned released evidence.

Choose a new output directory; do not overwrite frozen `results/`. Exact historical fitted-model inference is a different task: model weights, preprocessing artefacts and the complete author-side operational archive are not included. Recomputed numerical outputs do not reproduce measured historical wall-clock costs or turn the release into an independent training replication.

## Independent training

Obtain the raw datasets first. All new runs must use a new output root and explicit manifest. The following commands fit fresh baselines, then the six non-baseline interventions, and derive threshold studies from saved validation evidence:

```powershell
python src/main.py --dataset ulb --models all --strategy none --n_trials 50 --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main.py --dataset baf_base --models all --strategy none --n_trials 50 --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main.py --dataset ulb --models logreg rf lgbm catboost --strategy all --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main.py --dataset baf_base --models logreg rf lgbm catboost --strategy all --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main_transformer.py --dataset ulb --strategy none --n_trials 50 --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main_transformer.py --dataset baf_base --strategy none --n_trials 50 --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main_transformer.py --dataset ulb --strategy all --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/main_transformer.py --dataset baf_base --strategy all --results-root runs/reproduction --run-manifest runs/reproduction/revision_manifest.json
python src/threshold_study.py --dataset ulb --manifest runs/reproduction/revision_manifest.json --output-dir runs/reproduction/derived/thresholds/ulb_2013
python src/threshold_study.py --dataset baf_base --manifest runs/reproduction/revision_manifest.json --output-dir runs/reproduction/derived/thresholds/baf_base
```

Classical `--models all` includes OCSVM, not FT-Transformer. `--strategy all` runs the six interventions, not a new baseline. These fresh BAF searches differ from the preserved historical BAF selections underlying the manuscript; they are an independent rerun, not a bitwise replay. Representation-control and original correction queues have explicit source requirements and must not be treated as generic one-command reproductions of omitted archived models.

Synthetic tests do not retrain the full study:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

## Related work and citation

This paper shares corrected baseline and imbalance evidence with [Article 1](https://github.com/Applied-Intelligence-Hub/transformers-vs-ml-fraud-detection), which focuses on sensitivity, controlled transfer and interpretation boundaries. The [main thesis repository](https://github.com/Olavo200100274/fraud-imbalance-classical-vs-transformers) covers the broader dissertation and preserves its development history. These are related analyses, not independent replications.

> Caixeiro, O. M. C., Abbasi, M., and Martins, P. M. O. (2026). *Threshold Selection as a Critical Design Choice in Financial Fraud Detection: A Systematic Comparison of Classical Machine Learning Models and Tabular Transformers*. Manuscript in preparation for submission.

Include this repository URL and the commit used when citing code or evidence. Cite dataset authors and third-party methods separately. Do not invent journal acceptance, publication metadata or a DOI.

## AI assistance and rights

OpenAI Codex assisted with manuscript revision, checks against saved experimental evidence, plotting and layout code. Experimental values come from recorded evaluations, not generated scientific results. The authors remain responsible for accuracy and interpretation.

The code is provided under the repository's existing [GNU GPL version 3 licence](LICENSE). Dataset rights and terms remain with their providers, and the manuscript and third-party materials retain their respective rights; the code licence is not blanket permission to redistribute those materials.
