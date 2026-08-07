"""
==============================================================
Test Feature Engineering
==============================================================
"""

from pathlib import Path

import pandas as pd

from src.preprocessing.feature_engineering import FeatureEngineer


DATA_PATH = Path("data/raw/telemetry.csv")


def main():

    print()
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print()

    print("Preparing timestamps...")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print()

    print("Running Feature Engineering...")

    engineer = FeatureEngineer(df)

    features = engineer.run()

    print()

    print("=" * 75)
    print("FEATURE ENGINEERING")
    print("=" * 75)

    columns = [

        "timestamp",
        "generator_id",

        "fuel_level_l",
        "fuel_delta",

        "current",
        "current_delta",

        "battery_voltage",
        "voltage_delta",

        "time_delta_sec",

        "fuel_rate_lps",
        "fuel_rate_lph",

        "hour",
        "weekday",
        "is_weekend"

    ]

    print(features[columns].head(30))

    print()

    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)

    summary = pd.DataFrame({

        "Feature": [

            "Rows",
            "Columns",
            "Generators",
            "Average Fuel Rate (L/H)",
            "Maximum Fuel Rate (L/H)",
            "Minimum Fuel Rate (L/H)"

        ],

        "Value": [

            len(features),

            len(features.columns),

            features["generator_id"].nunique(),

            round(features["fuel_rate_lph"].mean(skipna=True), 3),

            round(features["fuel_rate_lph"].max(skipna=True), 3),

            round(features["fuel_rate_lph"].min(skipna=True), 3)

        ]

    })

    print(summary)

    print()

    output = Path("data/processed")

    output.mkdir(parents=True, exist_ok=True)

    features.to_csv(

        output / "feature_engineered_dataset.csv",

        index=False

    )

    print("Generated:")
    print("- feature_engineered_dataset.csv")


if __name__ == "__main__":

    main()