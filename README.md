# ml-fraud-threshold-selection-benchmark

This repository contains the code and benchmark evidence for evaluating how threshold selection and class imbalance strategies impact operational financial fraud detection. The suite demonstrates how to separate a model's ranking performance from its operational alert decisions, ensuring that operating points are selected entirely within development data.

**Key Features**

* **Algorithm Comparison:** Evaluates five supervised models (Logistic Regression, Random Forest, LightGBM, CatBoost, FT-Transformer) and a One-Class SVM reference.


* **Threshold Evaluation:** Implements validation-only operating rules (fixed 0.5, max-$F_{1}$, max-$F_{2}$, and a precision target) to quantify the trade-off between fraud recall and investigator alert workload.


* **Imbalance Interventions:** Includes a systematic comparison of seven class-imbalance strategies (None, RUS, ROS, SMOTE, SMOTE+Tomek, SMOTEENN, and Class Weights) applied across classical and transformer-based pipelines.


* **Datasets:** Provides the exact preprocessing, deduplication, and split workflows for the ULB 2013 credit card dataset and the Bank Account Fraud (BAF) Base dataset.



This codebase supports the generation of auditable, evidence-bounded configuration guidance rather than relying on a fixed 0.5 decision threshold.
