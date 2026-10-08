# Dataset inputs for Article 2

Raw CSVs are not distributed. Obtain the original provider files and follow
their access, attribution and licence terms.

| Input | Provider | Local filename |
|---|---|---|
| ULB credit-card transactions | [ULB on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) | `creditcard_2013.csv` (rename the downloaded `creditcard.csv`) |
| BAF Base | [Feedzai documentation](https://github.com/feedzai/bank-account-fraud), [Kaggle download](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) | `Base.csv` |

Place the decompressed files here. [manifest.json](manifest.json) identifies
the study's exact raw bytes; it also retains shared-suite hashes for provenance.
The five BAF Variants are not required for this manuscript's experiments.

```powershell
python src/verify_datasets.py --files creditcard_2013.csv Base.csv
```

Preserve CSV contents, formatting and line endings. The script verifies byte
sizes and SHA-256 hashes, not an independent assessment of dataset authenticity.
Dataset rights remain with the providers; GPL-3.0 for the code is not a new
licence for their datasets.

Raw inputs are needed for fresh training. Neither release integrity checks nor
numerical recomputation from the distributed frozen scores require downloading
datasets. ULB deduplication occurs before splitting; BAF months are pooled and
thresholds are selected within DEV, not on the held-out TEST labels.
