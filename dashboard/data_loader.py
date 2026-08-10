import pandas as pd

from config import DATA_PATH


def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset missing: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False
    )


    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )


    return df