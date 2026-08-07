"""
==============================================================
Test Generator State Estimator
==============================================================

Runs generator state estimation on telemetry dataset.

Input:
    data/processed/feature_engineered_dataset.csv

Output:
    data/processed/generator_state_estimated.csv

Author:
Fuel Telemetry AI Project
"""

from pathlib import Path

import pandas as pd


from src.preprocessing.generator_state_estimator import (
    GeneratorStateEstimator,
)



# ==========================================================
# Configuration
# ==========================================================


INPUT_FILE = (
    "data/processed/"
    "feature_engineered_dataset.csv"
)


OUTPUT_FILE = (
    "data/processed/"
    "generator_state_estimated.csv"
)



# ==========================================================
# Main Test
# ==========================================================


def main():

    print()

    print("=" * 70)

    print(
        "TESTING GENERATOR STATE ESTIMATOR"
    )

    print("=" * 70)



    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    print()

    print("Loading dataset...")


    if not Path(INPUT_FILE).exists():

        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )


    df = pd.read_csv(
        INPUT_FILE
    )


    print()

    print("Dataset Shape")

    print(df.shape)



    print()

    print("Columns")

    print(
        list(df.columns)
    )



    # ------------------------------------------------------
    # Timestamp preparation
    # ------------------------------------------------------

    if "timestamp" in df.columns:

        print()

        print(
            "Preparing timestamps..."
        )


        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )


    # ------------------------------------------------------
    # Run Estimator
    # ------------------------------------------------------

    print()

    print(
        "Running Generator State Estimator..."
    )


    estimator = GeneratorStateEstimator(
        df
    )


    result = estimator.run(
        save=False
    )



    # ------------------------------------------------------
    # Output preview
    # ------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "SAMPLE RESULTS"
    )

    print("=" * 70)



    preview_columns = [

        "generator_id",

        "status",

        "current",

        "battery_voltage",

        "estimated_status",

        "running_probability",

        "estimated_confidence",

        "confidence_level",

        "status_conflict",

    ]


    available_columns = [

        col

        for col in preview_columns

        if col in result.columns

    ]



    print(
        result[
            available_columns
        ]

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