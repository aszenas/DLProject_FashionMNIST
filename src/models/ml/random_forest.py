"""Random Forest classifier for Fashion-MNIST."""

from sklearn.ensemble import RandomForestClassifier

RANDOM_SEED = 42

BASELINE_PARAMS = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "class_weight": None,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


def build_random_forest(**overrides) -> RandomForestClassifier:
    """Create a Random Forest classifier with configurable parameters."""

    params = {**BASELINE_PARAMS, **overrides}

    return RandomForestClassifier(**params)
