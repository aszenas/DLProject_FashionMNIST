"""Tests for the Logistic Regression model builder."""

import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.models.ml.logistic_regression import (
    build_logistic_regression,
)


class TestLogisticRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_classification(
            n_samples=200,
            n_features=20,
            n_informative=15,
            n_redundant=0,
            n_classes=10,
            n_clusters_per_class=1,
            random_state=42,
        )

    def test_baseline_configuration(self):
        model = build_logistic_regression()

        self.assertIsInstance(model, LogisticRegression)
        self.assertEqual(model.C, 1.0)
        self.assertEqual(model.penalty, "l2")
        self.assertEqual(model.solver, "lbfgs")
        self.assertEqual(model.max_iter, 1000)
        self.assertEqual(model.random_state, 42)
        self.assertTrue(model.fit_intercept)
        self.assertIsNone(model.class_weight)

    def test_custom_configuration(self):
        model = build_logistic_regression(
            C=0.1,
            max_iter=500,
            seed=123,
        )

        self.assertEqual(model.C, 0.1)
        self.assertEqual(model.max_iter, 500)
        self.assertEqual(model.random_state, 123)

    def test_training_and_prediction(self):
        model = build_logistic_regression()
        model.fit(self.X, self.y)

        predictions = model.predict(self.X)

        self.assertEqual(predictions.shape, (200,))
        self.assertTrue(np.all(np.isin(predictions, np.arange(10))))

    def test_prediction_probabilities(self):
        model = build_logistic_regression()
        model.fit(self.X, self.y)

        probabilities = model.predict_proba(self.X)

        self.assertEqual(probabilities.shape, (200, 10))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.all(probabilities <= 1))

        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            np.ones(200),
            atol=1e-6,
        )

        np.testing.assert_array_equal(
            model.predict(self.X),
            model.classes_[probabilities.argmax(axis=1)],
        )

    def test_parameter_dimensions(self):
        model = build_logistic_regression()
        model.fit(self.X, self.y)

        self.assertEqual(model.coef_.shape, (10, 20))
        self.assertEqual(model.intercept_.shape, (10,))

    def test_independent_model_instances(self):
        model_a = build_logistic_regression()
        model_b = build_logistic_regression()

        self.assertIsNot(model_a, model_b)

        model_a.fit(self.X, self.y)

        self.assertTrue(hasattr(model_a, "coef_"))
        self.assertFalse(hasattr(model_b, "coef_"))

    def test_incompatible_multinomial_solver(self):
        with self.assertRaises(ValueError):
            build_logistic_regression(solver="liblinear")


if __name__ == "__main__":
    unittest.main()
