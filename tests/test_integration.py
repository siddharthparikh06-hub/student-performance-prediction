"""
End-to-End Integration Tests for Student Performance Prediction System
Module: tests/test_integration.py

Verifies the complete integration workflow from ML training artifacts
to Flask API endpoints, health reporting, and prediction responses.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from app import REQUIRED_FEATURES, create_app
from config import Config
from ml.train_model import train_and_evaluate


class TestEndToEndIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Build an isolated environment with synthetic dataset and trained artifacts."""
        cls.test_dir = tempfile.mkdtemp()
        cls.models_dir = os.path.join(cls.test_dir, "models")
        os.makedirs(cls.models_dir, exist_ok=True)

        # Generate a small valid synthetic dataset
        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "G1": np.random.uniform(5, 18, size=n),
            "G2": np.random.uniform(5, 19, size=n),
            "studytime": np.random.randint(1, 5, size=n),
            "failures": np.random.randint(0, 4, size=n),
            "absences": np.random.randint(0, 20, size=n),
            "G3": np.random.uniform(5, 20, size=n),
        })
        cls.dataset_path = os.path.join(cls.test_dir, "student_data.csv")
        df.to_csv(cls.dataset_path, index=False)

        # Train models into test_models_dir
        train_and_evaluate(
            data_path=cls.dataset_path,
            models_dir=cls.models_dir,
            test_size=0.2,
            random_state=42,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up isolated test files."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_pipeline_to_api_flow(self):
        """Verifies that trained artifacts integrate seamlessly with all API endpoints."""
        app = create_app({"TESTING": True, "MODELS_DIR": self.models_dir})
        client = app.test_client()

        # 1. Health Endpoint
        health_resp = client.get("/api/health")
        self.assertEqual(health_resp.status_code, 200)
        health_data = health_resp.get_json()
        self.assertEqual(health_data["status"], "healthy")
        self.assertTrue(health_data["models_available"])
        self.assertTrue(health_data["metrics_available"])
        self.assertTrue(health_data["models"]["linear_regression"])
        self.assertTrue(health_data["models"]["svr"])

        # 2. Metrics Endpoint
        metrics_resp = client.get("/api/metrics")
        self.assertEqual(metrics_resp.status_code, 200)
        metrics_data = metrics_resp.get_json()
        self.assertIn("models", metrics_data)
        self.assertIn("linear_regression", metrics_data["models"])
        self.assertIn("svr", metrics_data["models"])

        # 3. Predict Endpoint
        predict_resp = client.post(
            "/api/predict",
            json={
                "G1": 14.0,
                "G2": 15.0,
                "studytime": 3.0,
                "failures": 0.0,
                "absences": 2.0,
            },
        )
        self.assertEqual(predict_resp.status_code, 200)
        pred_data = predict_resp.get_json()
        self.assertIn("predictions", pred_data)
        self.assertIn("linear_regression", pred_data["predictions"])
        self.assertIn("svr", pred_data["predictions"])
        self.assertIsInstance(pred_data["predictions"]["linear_regression"], float)
        self.assertIsInstance(pred_data["predictions"]["svr"], float)

        # 4. Web Dashboard Route
        index_resp = client.get("/")
        self.assertEqual(index_resp.status_code, 200)

    def test_live_production_models_if_present(self):
        """Verifies default app configuration with real models/ folder."""
        models_dir = Path(Config.MODELS_DIR)
        lr_file = models_dir / "linear_regression.joblib"
        svr_file = models_dir / "svr.joblib"
        metrics_file = models_dir / "metrics.json"

        if lr_file.is_file() and svr_file.is_file() and metrics_file.is_file():
            app = create_app({"TESTING": True})
            client = app.test_client()

            health_resp = client.get("/api/health")
            self.assertEqual(health_resp.status_code, 200)
            self.assertEqual(health_resp.get_json()["status"], "healthy")

            metrics_resp = client.get("/api/metrics")
            self.assertEqual(metrics_resp.status_code, 200)
            metrics_data = metrics_resp.get_json()
            self.assertEqual(metrics_data["target"], "G3")

            predict_resp = client.post(
                "/api/predict",
                json={
                    "G1": 13.0,
                    "G2": 14.0,
                    "studytime": 2.0,
                    "failures": 0.0,
                    "absences": 4.0,
                },
            )
            self.assertEqual(predict_resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
