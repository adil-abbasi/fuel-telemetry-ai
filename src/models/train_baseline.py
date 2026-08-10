from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ======================================================
# PATH
# ======================================================

TEST_PATH = Path(
    "data/processed/test_fuel_forecasting.csv"
)


# ======================================================
# MAIN
# ======================================================

def main():

    print("\n" + "=" * 70)
    print("3-HOUR FUEL FORECASTING BASELINE")
    print("=" * 70)

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    print("\nLoading test dataset...")

    df = pd.read_csv(
        TEST_PATH,
        low_memory=False,
    )

    print(
        f"Test rows: {len(df):,}"
    )

    # --------------------------------------------------
    # Required columns
    # --------------------------------------------------

    required = [
        "fuel_level_l",
        "target_fuel_3h",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------
    # Valid rows
    # --------------------------------------------------

    valid = (
        df["fuel_level_l"].notna()
        &
        df["target_fuel_3h"].notna()
    )

    df = df.loc[valid].copy()

    print(
        f"Rows with current fuel: "
        f"{len(df):,}"
    )

    # --------------------------------------------------
    # Baseline prediction
    # --------------------------------------------------

    y_true = df[
        "target_fuel_3h"
    ]

    y_pred = df[
        "fuel_level_l"
    ]

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\nBaseline Results")

    print(
        f"MAE:  {mae:.2f} L"
    )

    print(
        f"RMSE: {rmse:.2f} L"
    )

    print(
        f"R²:   {r2:.4f}"
    )

    print("\nInterpretation")

    print(
        "The baseline assumes that the current "
        "fuel level remains unchanged for the "
        "next 3 hours."
    )

    print("\n" + "=" * 70)
    print("BASELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()