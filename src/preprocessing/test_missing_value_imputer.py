"""
==============================================================
Test - Missing Value Imputer
==============================================================

Tests intelligent telemetry missing value handling.

Input
-----
data/processed/generator_state_estimated.csv

Output
------
reports/
    missing_value_imputed_dataset.csv
    missing_value_summary.csv

Author:
Fuel Telemetry AI Project
"""


from pathlib import Path

import pandas as pd

from src.preprocessing.missing_value_imputer import (
    MissingValueImputer,
)


# ==========================================================
# Paths
# ==========================================================

DATA_PATH = Path(
    "data/processed/generator_state_estimated.csv"
)


REPORT_PATH = "reports"



# ==========================================================
# Main Test
# ==========================================================

def main():

    print()

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)


    print()

    print("Preparing timestamps...")


    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )


    print()

    print("Dataset Shape")

    print(df.shape)


    print()

    print("Missing Values Before")

    print(
        df[
            [
                "fuel_level_l",
                "current",
                "battery_voltage",
                "status",
            ]
        ]
        .isna()
        .sum()
    )


    print()

    print("Running Missing Value Imputer...")


    imputer = MissingValueImputer(
        df
    )


    result = imputer.run()



    print()

    print("=" * 70)

    print(
        "FINAL OUTPUT CHECK"
    )

    print("=" * 70)



    print()

    print(
        result.head(20)
    )



    print()

    print(
        "Columns Added:"
    )


    new_columns = [

        col

        for col in result.columns

        if col not in df.columns

    ]


    for col in new_columns:

        print(
            f" - {col}"
        )



    print()

    print(
        "Missing Values After"
    )


    print(

        result[
            [
                "fuel_level_l",
                "current",
                "battery_voltage",
                "status",
            ]
        ]

        .isna()

        .sum()

    )



    print()

    print(
        "Confidence Distribution"
    )


    if (
        "imputation_confidence_level"
        in result.columns
    ):

        print(

            result[
                "imputation_confidence_level"
            ]

            .value_counts()

            .sort_index()

        )



    print()

    print(
        "Saving Reports..."
    )


    imputer.save_reports(
        REPORT_PATH
    )


    print()

    print(
        "Generated:"
    )


    print(
        "- reports/missing_value_imputed_dataset.csv"
    )

    print(
        "- reports/missing_value_summary.csv"
    )



# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()