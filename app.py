"""
Student Performance & College CGPA Prediction - Flask Backend Application
Module: app.py

Provides web and API endpoints for:
1. School academic score prediction (UCI dataset standard, target G3).
2. College CGPA prediction (Study_Hours to CGPA regression).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd

from config import Config

# --- School Prediction Configuration ---
REQUIRED_FEATURES = ["G1", "G2", "studytime", "failures", "absences"]

FEATURE_CONSTRAINTS = {
    "G1": {"min": 0.0, "max": 20.0, "label": "First period grade (0-20)"},
    "G2": {"min": 0.0, "max": 20.0, "label": "Second period grade (0-20)"},
    "studytime": {"min": 1.0, "max": 4.0, "label": "Weekly study time (1-4)"},
    "failures": {"min": 0.0, "max": 4.0, "label": "Past class failures (0-4)"},
    "absences": {"min": 0.0, "max": 100.0, "label": "Number of absences (0-100)"},
}

# --- College Prediction Configuration ---
COLLEGE_REQUIRED_FEATURES = ["Study_Hours"]
COLLEGE_OPTIONAL_METADATA = ["student_name", "student_id", "department", "semester"]


def load_artifacts(models_dir: Path) -> Dict[str, Any]:
    """Loads model pipelines and metrics JSON from the specified directory."""
    artifacts = {
        "linear_regression": None,
        "svr": None,
        "metrics": None,
    }

    lr_path = models_dir / "linear_regression.joblib"
    svr_path = models_dir / "svr.joblib"
    metrics_path = models_dir / "metrics.json"

    if lr_path.is_file():
        try:
            artifacts["linear_regression"] = joblib.load(lr_path)
        except Exception:
            artifacts["linear_regression"] = None

    if svr_path.is_file():
        try:
            artifacts["svr"] = joblib.load(svr_path)
        except Exception:
            artifacts["svr"] = None

    if metrics_path.is_file():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                artifacts["metrics"] = json.load(f)
        except Exception:
            artifacts["metrics"] = None

    return artifacts


def validate_prediction_input(payload: Any) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """
    Validates school prediction payload.
    Ensures JSON object format, presence of all required features,
    numeric data types, and value range constraints.
    """
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object."

    missing = [feat for feat in REQUIRED_FEATURES if feat not in payload]
    if missing:
        return None, f"Missing required feature(s): {', '.join(missing)}."

    validated_values: Dict[str, float] = {}

    for feat in REQUIRED_FEATURES:
        val = payload[feat]

        if isinstance(val, bool) or val is None:
            return None, f"Feature '{feat}' must be a valid numeric value."

        try:
            numeric_val = float(val)
        except (ValueError, TypeError):
            return None, f"Feature '{feat}' must be a valid number, got: {val}."

        constraints = FEATURE_CONSTRAINTS[feat]
        if not (constraints["min"] <= numeric_val <= constraints["max"]):
            return None, (
                f"Feature '{feat}' value {numeric_val} is out of allowable range. "
                f"Allowed: [{constraints['min']}, {constraints['max']}] ({constraints['label']})."
            )

        validated_values[feat] = numeric_val

    return validated_values, None


def validate_college_prediction_input(
    payload: Any,
) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, str]], Optional[str]]:
    """
    Validates college CGPA prediction payload.
    Ensures JSON object format, presence of Study_Hours,
    numeric data types, and realistic range [0.0, 24.0] hours/day.
    Captures optional metadata without using it as predictive features.
    """
    if not isinstance(payload, dict):
        return None, None, "Request body must be a JSON object."

    if "Study_Hours" not in payload:
        return None, None, "Missing required feature: 'Study_Hours'."

    val = payload["Study_Hours"]
    if isinstance(val, bool) or val is None:
        return None, None, "Feature 'Study_Hours' must be a valid numeric value."

    try:
        study_hours = float(val)
    except (ValueError, TypeError):
        return None, None, f"Feature 'Study_Hours' must be a valid number, got: {val}."

    if not (0.0 <= study_hours <= 24.0):
        return None, None, f"Daily study hours must be between 0.0 and 24.0, got: {study_hours}."

    # Extract optional metadata
    metadata = {}
    for meta_key in COLLEGE_OPTIONAL_METADATA:
        if meta_key in payload and payload[meta_key] is not None:
            metadata[meta_key] = str(payload[meta_key]).strip()

    return {"Study_Hours": study_hours}, metadata, None


def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    """Application factory for Flask backend."""
    app = Flask(__name__)
    app.config.from_object(Config)

    if test_config:
        app.config.update(test_config)

    models_dir = Path(app.config.get("MODELS_DIR", Config.MODELS_DIR)).resolve()
    college_models_dir = Path(app.config.get("COLLEGE_MODELS_DIR", Config.COLLEGE_MODELS_DIR)).resolve()

    # -------------------------------------------------------------------------
    # Frontend Route
    # -------------------------------------------------------------------------
    @app.route("/")
    def index():
        try:
            return render_template("base.html")
        except Exception:
            return jsonify({
                "status": "online",
                "message": "Student Performance & College CGPA Prediction API is running."
            }), 200

    # -------------------------------------------------------------------------
    # School API Endpoints
    # -------------------------------------------------------------------------
    @app.route("/api/health", methods=["GET"])
    def health():
        artifacts = load_artifacts(models_dir)
        lr_loaded = artifacts["linear_regression"] is not None
        svr_loaded = artifacts["svr"] is not None
        metrics_loaded = artifacts["metrics"] is not None

        models_available = lr_loaded and svr_loaded
        status = "healthy" if (models_available and metrics_loaded) else "degraded"

        return jsonify({
            "status": status,
            "models_available": models_available,
            "metrics_available": metrics_loaded,
            "models": {
                "linear_regression": lr_loaded,
                "svr": svr_loaded,
            },
        }), 200

    @app.route("/api/metrics", methods=["GET"])
    def get_metrics():
        artifacts = load_artifacts(models_dir)
        metrics = artifacts["metrics"]

        if metrics is None:
            return jsonify({
                "error": "Model evaluation metrics are not available. Please train models first."
            }), 503

        return jsonify(metrics), 200

    @app.route("/api/predict", methods=["POST"])
    def predict():
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json."}), 400

        try:
            payload = request.get_json(silent=True)
        except Exception:
            return jsonify({"error": "Malformed JSON in request body."}), 400

        if payload is None:
            return jsonify({"error": "Invalid or empty JSON payload."}), 400

        validated_values, error_msg = validate_prediction_input(payload)
        if error_msg:
            return jsonify({"error": error_msg}), 400

        artifacts = load_artifacts(models_dir)
        lr_model = artifacts["linear_regression"]
        svr_model = artifacts["svr"]

        if lr_model is None or svr_model is None:
            return jsonify({
                "error": "Trained models are not available. Please ensure models are trained and saved."
            }), 503

        try:
            input_df = pd.DataFrame([validated_values], columns=REQUIRED_FEATURES)
            lr_pred = float(lr_model.predict(input_df)[0])
            svr_pred = float(svr_model.predict(input_df)[0])

            return jsonify({
                "features": validated_values,
                "predictions": {
                    "linear_regression": round(lr_pred, 2),
                    "svr": round(svr_pred, 2),
                },
            }), 200
        except Exception as exc:
            return jsonify({
                "error": f"Model inference failed: {str(exc)}"
            }), 500

    # -------------------------------------------------------------------------
    # College CGPA API Endpoints
    # -------------------------------------------------------------------------
    @app.route("/api/college/health", methods=["GET"])
    def college_health():
        artifacts = load_artifacts(college_models_dir)
        lr_loaded = artifacts["linear_regression"] is not None
        svr_loaded = artifacts["svr"] is not None
        metrics_loaded = artifacts["metrics"] is not None

        models_available = lr_loaded and svr_loaded
        status = "healthy" if (models_available and metrics_loaded) else "degraded"

        return jsonify({
            "status": status,
            "models_available": models_available,
            "metrics_available": metrics_loaded,
            "models": {
                "linear_regression": lr_loaded,
                "svr": svr_loaded,
            },
        }), 200

    @app.route("/api/college/metrics", methods=["GET"])
    def get_college_metrics():
        artifacts = load_artifacts(college_models_dir)
        metrics = artifacts["metrics"]

        if metrics is None:
            return jsonify({
                "error": "College model evaluation metrics are not available. Please train college models first."
            }), 503

        return jsonify(metrics), 200

    @app.route("/api/college/predict", methods=["POST"])
    def college_predict():
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json."}), 400

        try:
            payload = request.get_json(silent=True)
        except Exception:
            return jsonify({"error": "Malformed JSON in request body."}), 400

        if payload is None:
            return jsonify({"error": "Invalid or empty JSON payload."}), 400

        validated_values, metadata, error_msg = validate_college_prediction_input(payload)
        if error_msg:
            return jsonify({"error": error_msg}), 400

        artifacts = load_artifacts(college_models_dir)
        lr_model = artifacts["linear_regression"]
        svr_model = artifacts["svr"]

        if lr_model is None or svr_model is None:
            return jsonify({
                "error": "Trained college models are not available. Please ensure college models are trained and saved."
            }), 503

        try:
            input_df = pd.DataFrame([validated_values], columns=COLLEGE_REQUIRED_FEATURES)
            raw_lr_pred = float(lr_model.predict(input_df)[0])
            raw_svr_pred = float(svr_model.predict(input_df)[0])

            # Ensure displayed predictions remain within the valid CGPA range of 0.0 to 10.0
            lr_pred = max(0.0, min(10.0, raw_lr_pred))
            svr_pred = max(0.0, min(10.0, raw_svr_pred))

            response_data = {
                "features": validated_values,
                "predictions": {
                    "linear_regression": round(lr_pred, 2),
                    "svr": round(svr_pred, 2),
                },
            }
            if metadata:
                response_data["metadata"] = metadata

            return jsonify(response_data), 200
        except Exception as exc:
            return jsonify({
                "error": f"College model inference failed: {str(exc)}"
            }), 500

    # -------------------------------------------------------------------------
    # Global Error Handlers
    # -------------------------------------------------------------------------
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Resource not found."}), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({"error": "Method not allowed for this endpoint."}), 405

    return app


app = create_app()

if __name__ == "__main__":
    import os
    # Render provides PORT; default to 5000 for local runs.
    port = int(os.getenv("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

