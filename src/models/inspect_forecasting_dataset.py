from pathlib import Path

import pandas as pd


# ======================================================
# PATH
# ======================================================

DATA_PATH = Path(
    "data/processed/imputed_telemetry_dataset.csv"
)


# ======================================================
# MAIN
# ======================================================

def main():

    print("\n" + "=" * 70)
    print("FORECASTING DATASET INSPECTION")
    print("=" * 70)

    # --------------------------------------------------
    # Load Dataset
    # --------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    print("\nDataset Shape:")
    print(df.shape)

    # --------------------------------------------------
    # Timestamp
    # --------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    print("\nTimestamp Information")

    print(
        "Minimum:",
        df["timestamp"].min()
    )

    print(
        "Maximum:",
        df["timestamp"].max()
    )

    print(
        "Missing timestamps:",
        df["timestamp"].isna().sum()
    )

    # --------------------------------------------------
    # Generators
    # --------------------------------------------------

    print("\nGenerators:")

    print(
        df["generator_id"]
        .nunique()
    )

    print(
        df["generator_id"]
        .value_counts()
    )

    # --------------------------------------------------
    # Required Forecasting Columns
    # --------------------------------------------------

    required_columns = [
        "timestamp",
        "generator_id",
        "fuel_level_l",
        "fuel_rate_lph",
        "fuel_delta",
        "current",
        "current_delta",
        "battery_voltage",
        "voltage_delta",
        "estimated_status",
        "running_probability",
        "hour",
        "minute",
        "weekday",
        "is_weekend",
        "time_delta_sec",
        "telemetry_quality_score",
    ]

    print("\nRequired Forecasting Columns")

    for column in required_columns:

        if column in df.columns:

            print(
                f"[OK]      {column}"
            )

        else:

            print(
                f"[MISSING] {column}"
            )

    # --------------------------------------------------
    # Data Types
    # --------------------------------------------------

    print("\nData Types")

    print(
        df[
            [
                column
                for column in required_columns
                if column in df.columns
            ]
        ].dtypes
    )

    # --------------------------------------------------
    # Missing Values
    # --------------------------------------------------

    print("\nMissing Values")

    missing = (
        df[
            [
                column
                for column in required_columns
                if column in df.columns
            ]
        ]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(missing)

    # --------------------------------------------------
    # Fuel Statistics
    # --------------------------------------------------

    print("\nFuel Statistics")

    print(
        df["fuel_level_l"]
        .describe()
        .round(2)
    )

    # --------------------------------------------------
    # Fuel Rate Statistics
    # --------------------------------------------------

    print("\nFuel Rate Statistics")

    print(
        df["fuel_rate_lph"]
        .describe()
        .round(4)
    )

    # --------------------------------------------------
    # Status Distribution
    # --------------------------------------------------

    print("\nEstimated Status Distribution")

    print(
        df["estimated_status"]
        .value_counts(
            dropna=False
        )
    )

    # --------------------------------------------------
    # Generator Time Coverage
    # --------------------------------------------------

    print("\nGenerator Time Coverage")

    coverage = (
        df.groupby("generator_id")
        .agg(
            start_time=(
                "timestamp",
                "min"
            ),
            end_time=(
                "timestamp",
                "max"
            ),
            records=(
                "timestamp",
                "count"
            ),
        )
    )

    coverage["duration_hours"] = (
        (
            coverage["end_time"]
            - coverage["start_time"]
        )
        .dt.total_seconds()
        / 3600
    )

    print(
        coverage.round(2)
    )

    # --------------------------------------------------
    # Final Information
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
