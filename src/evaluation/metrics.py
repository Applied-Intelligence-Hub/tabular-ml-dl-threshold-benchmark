import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    brier_score_loss,
    confusion_matrix,
)


# ─────────────────────────────────────────────────────────────────────────
# Core metric functions
# ─────────────────────────────────────────────────────────────────────────

def compute_pr_auc(y_true, y_scores):
    """Area under the Precision-Recall curve (average_precision_score)."""
    return average_precision_score(y_true, y_scores)


def find_threshold_maximizing_f2(y_true, y_scores):
    """
    Find the decision threshold that maximises the F2 score.

    Returns
    -------
    best_threshold : float
    best_f2 : float
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # precision_recall_curve returns (n+1) precision/recall values and n thresholds.
    p = precisions[:-1]
    r = recalls[:-1]

    beta = 2.0
    numerator = (1 + beta**2) * p * r
    denominator = (beta**2 * p) + r
    with np.errstate(divide="ignore", invalid="ignore"):
        f2_scores = np.where(denominator > 0, numerator / denominator, 0.0)

    best_idx = np.argmax(f2_scores)
    return float(thresholds[best_idx]), float(f2_scores[best_idx])


def find_threshold_maximizing_f1(y_true, y_scores):
    """
    Find the decision threshold that maximises the F1 score.

    Returns
    -------
    best_threshold : float
    best_f1 : float
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    p = precisions[:-1]
    r = recalls[:-1]

    numerator = 2 * p * r
    denominator = p + r
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(denominator > 0, numerator / denominator, 0.0)

    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def find_threshold_at_min_precision(y_true, y_scores, min_precision=0.5):
    """
    Find the lowest threshold that ensures Precision >= min_precision.

    Sweeps thresholds from precision_recall_curve (sorted ascending) and
    returns the lowest one whose precision meets the constraint.

    Returns
    -------
    best_threshold : float   (or np.inf if no threshold satisfies the constraint)
    precision_at_threshold : float
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # precision_recall_curve returns thresholds in ascending order.
    # precisions[:-1] and recalls[:-1] correspond to thresholds.
    p = precisions[:-1]

    # Find indices where precision >= min_precision
    valid = np.where(p >= min_precision)[0]
    if len(valid) == 0:
        return float("inf"), 0.0

    # Lowest threshold that meets the constraint
    best_idx = valid[0]
    return float(thresholds[best_idx]), float(p[best_idx])


def compute_all_metrics(y_true, y_scores, threshold):
    """
    Compute every evaluation metric required by the thesis protocol.

    Returns a dict with:
        PR-AUC, F1, F2, TP, FP, TN, FN, alert_rate, FP/TP,
        threshold, brier_score, precision_at_k, recall_at_k
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    n = len(y_pred)
    alert_rate = int((y_pred == 1).sum()) / n if n > 0 else 0.0
    fp_tp_ratio = fp / tp if tp > 0 else float("inf")

    # ── Brier score (calibration quality) ─────────────────────────────
    # Requires probabilities in [0, 1]. If scores are not probabilities
    # (e.g. OCSVM decision_function), we clip for a best-effort Brier.
    y_scores_clipped = np.clip(y_scores, 0.0, 1.0)
    brier = brier_score_loss(y_true, y_scores_clipped)

    # ── Workload metrics: Precision@k and Recall@k ────────────────────
    prec_at_k, recall_at_k, k_used = _precision_recall_at_k(y_true, y_scores)

    return {
        "PR-AUC": round(float(pr_auc), 6),
        "ROC-AUC": round(float(roc_auc), 6),
        "F1": round(float(f1), 6),
        "F2": round(float(f2), 6),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "alert_rate": round(float(alert_rate), 6),
        "FP/TP": round(float(fp_tp_ratio), 4),
        "threshold": round(float(threshold), 6),
        "brier_score": round(float(brier), 6),
        "precision_at_k": round(float(prec_at_k), 6),
        "recall_at_k": round(float(recall_at_k), 6),
        "k_used": int(k_used),
    }


# ─────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────

def bootstrap_ci(y_true, y_scores, threshold, metric_fn=None,
                 n_bootstrap=1000, ci=0.95, random_state=42):
    """
    Compute bootstrap 95 % CI for PR-AUC and F2 on the test set.

    Returns
    -------
    dict with 'PR-AUC_ci' and 'F2_ci', each a (lower, upper) tuple.
    """
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n = len(y_true)

    pr_aucs = []
    roc_aucs = []
    f2s = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        ys = y_scores[idx]

        # Need both classes in the bootstrap sample
        if yt.sum() == 0 or yt.sum() == n:
            continue

        pr_aucs.append(average_precision_score(yt, ys))
        roc_aucs.append(roc_auc_score(yt, ys))
        y_pred = (ys >= threshold).astype(int)
        f2s.append(fbeta_score(yt, y_pred, beta=2, zero_division=0))

    alpha = (1 - ci) / 2
    return {
        "PR-AUC_ci": (
            round(float(np.percentile(pr_aucs, 100 * alpha)), 6),
            round(float(np.percentile(pr_aucs, 100 * (1 - alpha))), 6),
        ),
        "ROC-AUC_ci": (
            round(float(np.percentile(roc_aucs, 100 * alpha)), 6),
            round(float(np.percentile(roc_aucs, 100 * (1 - alpha))), 6),
        ),
        "F2_ci": (
            round(float(np.percentile(f2s, 100 * alpha)), 6),
            round(float(np.percentile(f2s, 100 * (1 - alpha))), 6),
        ),
    }


# ─────────────────────────────────────────────────────────────────────────
# Workload helpers
# ─────────────────────────────────────────────────────────────────────────

def _precision_recall_at_k(y_true, y_scores, top_fraction=0.005):
    """
    Precision and Recall in the top-k alerts (default: top 0.5 %).

    This simulates an investigator reviewing only the k highest-scored
    transactions and measures how many of those are actually fraud.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n = len(y_true)
    k = max(1, int(n * top_fraction))

    top_k_idx = np.argsort(y_scores)[::-1][:k]
    tp_at_k = y_true[top_k_idx].sum()
    total_positives = y_true.sum()

    precision_at_k = tp_at_k / k if k > 0 else 0.0
    recall_at_k = tp_at_k / total_positives if total_positives > 0 else 0.0

    return precision_at_k, recall_at_k, k
