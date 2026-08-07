"""
==============================================================
Test Telemetry Validator
==============================================================

Input:
data/processed/generator_state_estimated.csv

Output:
data/processed/validated_telemetry_dataset.csv

==============================================================
"""


from pathlib import Path

import pandas as pd


from src.preprocessing.telemetry_validator import (
    TelemetryValidator,
)



# ==========================================================
# Files
# ==========================================================


INPUT_FILE = (
    "data/processed/"
    "generator_state_estimated.csv"
)


OUTPUT_FILE = (
    "data/processed/"
    "validated_telemetry_dataset.csv"
)



# ==========================================================
# Main
# ==========================================================


def main():


    print()

    print("=" * 70)

    print(
        "TESTING TELEMETRY VALIDATOR"
    )

    print("=" * 70)



    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------


    print()

    print(
        "Loading dataset..."
    )


    if not Path(INPUT_FILE).exists():

        raise FileNotFoundError(

            f"File not found: {INPUT_FILE}"

        )


    df = pd.read_csv(

        INPUT_FILE

    )



    print()

    print(
        "Dataset Shape:"
    )

    print(
        df.shape
    )



    # ------------------------------------------------------
    # Prepare timestamp
    # ------------------------------------------------------


    print()

    print(
        "Preparing timestamps..."
    )


    df["timestamp"] = pd.to_datetime(

        df["timestamp"],

        errors="coerce"

    )



    # ------------------------------------------------------
    # Run Validator
    # ------------------------------------------------------


    validator = TelemetryValidator(

        df

    )


    result = validator.run()



    # ------------------------------------------------------
    # Report
    # ------------------------------------------------------


    print()

    print("=" * 70)

    print(
        "TELEMETRY VALIDATION REPORT"
    )

    print("=" * 70)



    print()


    print(
        "Rows Processed:",
        len(result)
    )



    print()


    print(
        "Fuel Invalid:"
    )

    print(

        result["fuel_invalid"]

        .sum()

    )



    print()


    print(
        "Fuel Outliers:"
    )

    print(

        result["fuel_outlier"]

        .sum()

    )



    print()


    print(
        "Current Invalid:"
    )

    print(

        result["current_invalid"]

        .sum()

    )



    print()


    print(
        "Current Outliers:"
    )

    print(

        result["current_outlier"]

        .sum()

    )



    print()


    print(
        "Battery Invalid:"
    )

    print(

        result["battery_invalid"]

        .sum()

    )



    print()


    print(
        "Timestamp Duplicates:"
    )

    print(

        result["timestamp_duplicate"]

        .sum()

    )



    print()


    print(
        "Timestamp Gaps:"
    )

    print(

        result["timestamp_gap"]

        .sum()

    )



    print()


    print(
        "Average Telemetry Quality:"
    )

    print(

        round(

            result[

                "telemetry_quality_score"

            ]

            .mean(),

            2

        ),

        "%"

    )



    # ------------------------------------------------------
    # Sample Results
    # ------------------------------------------------------


    print()

    print("=" * 70)

    print(
        "SAMPLE RESULTS"
    )

    print("=" * 70)



    columns = [

        "generator_id",

        "fuel_level_l",

        "current",

        "battery_voltage",

        "fuel_invalid",

        "fuel_outlier",

        "current_outlier",

        "battery_invalid",

        "telemetry_quality_score",

        "validation_reason",

        "battery_current_mismatch",

        "state_sensor_mismatch",

    ]



    available = [

        col

        for col in columns

        if col in result.columns

    ]



    print(

        result[available]

        .head(30)

        .to_string()

    )



    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------


    print()

    print(
        "Saving output..."
    )



    Path(

        OUTPUT_FILE

    ).parent.mkdir(

        parents=True,

        exist_ok=True

    )



    result.to_csv(

        OUTPUT_FILE,

        index=False

    )



    print()

    print("=" * 70)

    print(
        "GENERATED:"
    )

    print(
        OUTPUT_FILE
    )

    print("=" * 70)





if __name__ == "__main__":

    main()