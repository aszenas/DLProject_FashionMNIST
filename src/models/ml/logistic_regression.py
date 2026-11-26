"""Multinomial Logistic Regression for Fashion-MNIST."""

from sklearn.linear_model import LogisticRegression

DEFAULT_SEED = 42


def build_logistic_regression(
    *,
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    max_iter: int = 1000,
    tol: float = 1e-4,
    fit_intercept: bool = True,
    class_weight: str | dict | None = None,
    l1_ratio: float | None = None,
    seed: int = DEFAULT_SEED,
) -> LogisticRegression:
    """Create a configurable multinomial Logistic Regression model.

    The default configuration corresponds to the Fashion-MNIST
    baseline defined in the model research documentation.

    Parameters
    ----------
    C:
        Inverse regularization strength.
    penalty:
        Regularization type.
    solver:
        Optimization algorithm.
    max_iter:
        Maximum number of optimization iterations.
    tol:
        Optimization stopping tolerance.
    fit_intercept:
        Whether to learn an intercept.
    class_weight:
        Optional class weighting.
    l1_ratio:
        Elastic Net mixing parameter.
    seed:
        Random seed for solvers that use randomness.

    Returns
    -------
    LogisticRegression
        An unfitted Logistic Regression estimator.
    """
    if solver == "liblinear":
        raise ValueError("The liblinear solver does not support multinomial Logistic Regression.")

    return LogisticRegression(
        C=C,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        tol=tol,
        fit_intercept=fit_intercept,
        class_weight=class_weight,
        l1_ratio=l1_ratio,
        random_state=seed,
    )
