import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-student-marks")
    DATASET_DIR = BASE_DIR / "dataset"
    MODELS_DIR = BASE_DIR / "models"
    DATASET_PATH = (
        DATASET_DIR / "student_data.csv"
        if (DATASET_DIR / "student_data.csv").is_file()
        else DATASET_DIR / "student-mat.csv"
    )
