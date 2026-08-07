import pandas as pd

from config import RAW_DATA

from src.utils.data_loader import load_dataset

from src.preprocessing.missing_telemetry_handler import (
    MissingTelemetryHandler
)


def main():

    print()

    print("Loading dataset...")

    df = load_dataset(RAW_DATA)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    handler = MissingTelemetryHandler(df)

    processed_df, report = handler.run()

    print()

    print("=" * 60)

    print("MISSING TELEMETRY REPORT")

    print("=" * 60)

    print(report)

    print()

    print(processed_df[

        processed_df["is_estimated"]

    ][

        [

            "timestamp",

            "generator_id",

            "fuel_level_l",

            "fuel_level_l_clean",

            "current",

            "current_clean",

            "battery_voltage",

            "battery_voltage_clean",

            "gap_type"

        ]

    ].head(30))

if __name__ == "__main__":

    main()