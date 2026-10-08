"""
Imbalance handling strategies for the fraud detection pipeline.

Usage:
    from strategies.balancing import get_sampler, apply_class_weights

    sampler = get_sampler("smote")          # returns imblearn sampler
    X_res, y_res = sampler.fit_resample(X, y)

    pipeline = apply_class_weights(pipeline) # injects class_weight='balanced'
"""

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTETomek, SMOTEENN


# All valid strategy names (used for CLI validation)
STRATEGY_NAMES = [
    "none",
    "rus",
    "ros",
    "smote",
    "smote_tomek",
    "smoteenn",
    "weights",
]

# Strategies that use a sampler (data-level resampling)
RESAMPLING_STRATEGIES = {"rus", "ros", "smote", "smote_tomek", "smoteenn"}


def get_sampler(strategy_name, random_state=42):
    """
    Return an imblearn sampler for the given strategy.

    Parameters
    ----------
    strategy_name : str
        One of: 'rus', 'ros', 'smote', 'smote_tomek', 'smoteenn'.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    sampler : imblearn sampler instance

    Raises
    ------
    ValueError
        If strategy_name is not a resampling strategy.
    """
    if strategy_name == "rus":
        return RandomUnderSampler(random_state=random_state)
    elif strategy_name == "ros":
        return RandomOverSampler(random_state=random_state)
    elif strategy_name == "smote":
        return SMOTE(random_state=random_state)
    elif strategy_name == "smote_tomek":
        return SMOTETomek(random_state=random_state)
    elif strategy_name == "smoteenn":
        return SMOTEENN(random_state=random_state)
    else:
        raise ValueError(
            f"'{strategy_name}' is not a resampling strategy. "
            f"Valid options: {sorted(RESAMPLING_STRATEGIES)}"
        )


def apply_class_weights(pipeline):
    """
    Inject class_weight='balanced' into the classifier step of a pipeline.

    Supports: LogisticRegression, RandomForest, LGBMClassifier, CatBoostClassifier.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Must have a step named 'classifier'.

    Returns
    -------
    pipeline : same pipeline with class_weight set.
    """
    clf = pipeline.named_steps["classifier"]
    clf_name = type(clf).__name__

    if clf_name == "CatBoostClassifier":
        # CatBoost uses 'auto_class_weights' instead of 'class_weight'
        clf.set_params(auto_class_weights="Balanced")
    elif clf_name == "LGBMClassifier":
        clf.set_params(class_weight="balanced")
    else:
        # LogisticRegression, RandomForestClassifier
        clf.set_params(class_weight="balanced")

    return pipeline
