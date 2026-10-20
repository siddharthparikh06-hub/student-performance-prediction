"""
Unit and Integration Tests for Flask Backend
Module: tests/test_app.py

Tests Flask web and API routes, input validation, error handling,
and model-loading states using isolated fixtures and mocked artifacts.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from app import REQUIRED_FEATURES, create_app


class TestFlaskApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prepare temporary directory and synthetic model artifacts for testing."""
        cls.test_dir = tempfile.mkdtemp()
        cls.empty_models_dir = os.path.join(cls.test_dir, "empty_models")
        cls.loaded_models_dir = os.path.join(cls.test_dir, "loaded_models")
        os.makedirs(cls.empty_models_dir, exist_ok=True)
        os.makedirs(cls.loaded_models_dir, exist_ok=True)

        # Train minimal dummy pipelines
        np.random.seed(42)
        X_dummy = pd.DataFrame(
            np.random.uniform(5, 15, size=(20, len(REQUIRED_FEATURES))),
            columns=REQUIRED_FEATURES,
        )
        y_dummy = pd.Series(np.random.uniform(5, 15, size=20))

        lr_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ])
        lr_pipeline.fit(X_dummy, y_dummy)

        svr_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf")),
        ])
        svr_pipeline.fit(X_dummy, y_dummy)

        # Save dummy artifacts
        joblib.dump(lr_pipeline, os.path.join(cls.loaded_models_dir, "linear_regression.joblib"))
        joblib.dump(svr_pipeline, os.path.join(cls.loaded_models_dir, "svr.joblib"))

        cls.sample_metrics = {
            "features": REQUIRED_FEATURES,
            "target": "G3",
            "split_settings": {"test_size": 0.2, "random_state": 42},
            "models": {
                "linear_regression": {"metrics": {"mae": 1.25, "rmse": 1.55, "r2": 0.85}},
                "svr": {"metrics": {"mae": 1.15, "rmse": 1.45, "r2": 0.88}},
            },
        }

        with open(os.path.join(cls.loaded_models_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(cls.sample_metrics, f)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test artifacts."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_index_route(self):
        """Test that the index route renders successfully with dashboard elements."""
        app = create_app({"TESTING": True, "MODELS_DIR": self.empty_models_dir})
        client = app.test_client()

        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Student Performance Prediction", response.data)
        self.assertIn(b"predictionForm", response.data)
        self.assertIn(b"modelComparisonChart", response.data)
        self.assertIn(b"sessionTable", response.data)

    def test_health_endpoint_without_models(self):
        """Test health endpoint reports degraded status when models are missing."""
        app = create_app({"TESTING": True, "MODELS_DIR": self.empty_models_dir})
        client = app.test_client()

        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "degraded")
        self.assertFalse(data["models_available"])
        self.assertFalse(data["metrics_available"])

    def test_metrics_endpoint_without_models(self):
        """Test metrics endpoint returns 503 when metrics.json does not exist."""
        app = create_app({"TESTING": True, "MODELS_DIR": self.empty_models_dir})
        client = app.test_client()

        response = client.get("/api/metrics")
        self.assertEqual(response.status_code, 503)
        data = response.get_json()
        self.assertIn("error", data)

    def test_predict_endpoint_without_models(self):
        """Test predict endpoint returns 503 when models are not loaded."""
        app = create_app({"TESTING": True, "MODELS_DIR": self.empty_models_dir})
        client = app.test_client()

        payload = {
            "G1": 14.0,
            "G2": 15.0,
            "studytime": 2.0,
            "failures": 0.0,
            "absences": 4.0,
        }
        response = client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 503)
        data = response.get_json()
        self.assertIn("error", data)

    def test_health_endpoint_with_models(self):
        """Test health endpoint reports healthy status when artifacts are present."""
        app = create_app({"TESTING": True, "MODELS_DIR": self.loaded_models_dir})
        client = app.test_client()

        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["models_available"])
        self.assertTrue(data["metrics_available"])
        self.assertTrue(data["models"]["linear_regression"])
        self.assertTrue(data["models"]["svr"])

    def test_metrics_endpoint_with_models(self):
        """Test metrics endpoint returns saved metrics successfully."""
        app = create_app({"TESTING": True, "MODELS_DIR": self.loaded_models_dir})
        client = app.test_client()

        response = client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["target"], "G3")
        self.assertIn("linear_regression", data["models"])
        self.assertIn("svr", data["models"])

    def test_predict_endpoint_valid_payload(self):
        """Test predict endpoint returns numeric predictions for both models."""
        app = create_app({"TESTING": True, "MODELS_DIR": self.loaded_models_dir})
        client = app.test_client()

        payload = {
            "G1": 12.0,
            "G2": 14.0,
            "studytime": 3.0,
            "failures": 0.0,
            "absences": 2.0,
        }
        response = client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn("predictions", data)
        self.assertIn("linear_regression", data["predictions"])
        self.assertIn("svr", data["predictions"])
        self.assertIsInstance(data["predictions"]["linear_regression"], float)
        self.assertIsInstance(data["predictions"]["svr"], float)

    def test_predict_endpoint_non_json(self):
        """Test predict endpoint returns 400 when request is not JSON."""
        app = create_app({"TESTING": True, "MODELS_DIR": self.loaded_models_dir})
        client = app.test_client()

        response = client.post("/api/predict", data="not a json", content_type="text/plain")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)

    def test_predict_endpoint_missing_feature(self):
        """Test predict endpoint returns 400 when a required feature is omitted."""
        app = create_app({"TESTING": True, "MODELS_DIR": self.loaded_models_dir})
        client = app.test_client()

        payload = {
            "G1": 12.0,
            "G2": 14.0,
            "studytime": 3.0,
            # 'failures' missing
            "absences": 2.0,
        }
        response = client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("Missing required feature(s)", data["error"])

    def test_predict_endpoint_non_numeric_value(self):
        """Test predict endpoint returns 400 when non-numeric feature is passed."""
        app = create_app({"TESTING": True, "MODELS_DIR": self.loaded_models_dir})
        client = app.test_client()

        payload = {
            "G1": "grade_a",
            "G2": 14.0,
            "studytime": 3.0,
            "failures": 0.0,
            "absences": 2.0,
        }
        response = client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("must be a valid number", data["error"])

    def test_predict_endpoint_out_of_range_value(self):
        """Test predict endpoint returns 400 when feature exceeds allowable bounds."""
        app = create_app({"TESTING": True, "MODELS_DIR": self.loaded_models_dir})
        client = app.test_client()

        payload = {
            "G1": 25.0,  # Max allowed is 20
            "G2": 14.0,
            "studytime": 3.0,
            "failures": 0.0,
            "absences": 2.0,
        }
        response = client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("out of allowable range", data["error"])

    def test_predict_endpoint_method_not_allowed(self):
        """Test GET request on /api/predict returns 405."""
        app = create_app({"TESTING": True, "MODELS_DIR": self.loaded_models_dir})
        client = app.test_client()

        response = client.get("/api/predict")
        self.assertEqual(response.status_code, 405)


if __name__ == "__main__":
    unittest.main()
