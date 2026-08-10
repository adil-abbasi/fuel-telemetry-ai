from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sensor_health_dataset.csv"
)