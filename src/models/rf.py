from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def get_pipeline_and_params(preprocessor):
    """Return (pipeline, suggest_fn) for Random Forest.

    Search space (Optuna):
    - n_estimators: [50, 500] step 50 — forest size.
    - max_depth: categorical {None, 10, 20, 30, 40, 50} — tree depth.
    - min_samples_split: [2, 20] — min samples to split an internal node.
    - min_samples_leaf: [1, 10] — min samples at a leaf.
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42)),
    ])

    def suggest_params(trial):
        return {
            "classifier__n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "classifier__max_depth": trial.suggest_categorical("max_depth", [None, 10, 20, 30, 40, 50]),
            "classifier__min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "classifier__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }

    return pipeline, suggest_params