from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent

# Data
RAW_DATA = BASE_DIR / "data" / "raw" / "telemetry.csv"

PROCESSED_DATA = BASE_DIR / "data" / "processed"

MODEL_PATH = BASE_DIR / "data" / "models"

REPORT_PATH = BASE_DIR / "reports"  