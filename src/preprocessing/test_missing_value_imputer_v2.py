"""
==============================================================
Test Missing Value Imputer V2
==============================================================

Input:
data/processed/validated_telemetry_dataset.csv

Output:
data/processed/imputed_telemetry_dataset.csv

==============================================================
"""

from pathlib import Path

import pandas as pd


from src.preprocessing.missing_value_imputer_v2 import (
    MissingValueImputerV2
)



# ==========================================================
# Files
# ==========================================================


INPUT_FILE = (

    "data/processed/"

    "validated_telemetry_dataset.csv"

)


OUTPUT_FILE = (

    "data/processed/"

    "imputed_telemetry_dataset.csv"

)



# ==========================================================
# Main
# ==========================================================


def main():


    print()

    print("=" * 70)

    print(

        "TESTING MISSING VALUE IMPUTER V2"

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

            INPUT_FILE

        )



    df = pd.read_csv(

        INPUT_FILE

    )



    print()

    print(

        "Dataset Shape"

    )

    print(

        df.shape

    )



    # ------------------------------------------------------
    # Missing Before
    # ------------------------------------------------------


    print()

    print("=" * 70)

    print(

        "MISSING VALUES BEFORE"

    )

    print("=" * 70)



    check_columns = [

        "fuel_level_l",

        "current",

        "battery_voltage",

        "status"

    ]



    print(

        df[check_columns]

        .isna()

        .sum()

    )



    # ------------------------------------------------------
    # Run Imputer
    # ------------------------------------------------------


    imputer = MissingValueImputerV2(

        df

    )



    result = imputer.run()



    # ------------------------------------------------------
    # Report
    # ------------------------------------------------------


    print()

    print("=" * 70)

    print(

        "IMPUTATION REPORT"

    )

    print("=" * 70)



    print()

    print(

        "Rows Processed:",

        len(result)

    )



    print()

    print(

        "Fuel Filled:"

    )

    print(

        result["fuel_imputed"]

        .sum()

    )



    print()

    print(

        "Current Filled:"

    )

    print(

        result["current_imputed"]

        .sum()

    )



    print()

    print(

        "Battery Filled:"

    )

    print(

        result["battery_imputed"]

        .sum()

    )



    print()

    print(

        "Status Filled:"

    )

    print(

        result["status_imputed"]

        .sum()

    )



    # ------------------------------------------------------
    # Missing After
    # ------------------------------------------------------


    print()

    print("=" * 70)

    print(

        "MISSING VALUES AFTER"

    )

    print("=" * 70)



    print(

        result[check_columns]

        .isna()

        .sum()

    )



    # ------------------------------------------------------
    # Confidence
    # ------------------------------------------------------


    print()

    print("=" * 70)

    print(

        "CONFIDENCE DISTRIBUTION"

    )

    print("=" * 70)



    print(

        result[

            "imputation_confidence_level"

        ]

        .value_counts()

    )



    print()

    print(

        "Average Confidence:"

    )


    print(

        round(

            result[

                "imputation_confidence"

            ]

            .mean(),

            2

        ),

        "%"

    )



    # ------------------------------------------------------
    # Sample
    # ------------------------------------------------------


    print()

    print("=" * 70)

    print(

        "SAMPLE RESULTS"

    )

    print("=" * 70)



    sample_columns = [

        "generator_id",

        "fuel_level_l",

        "current",

        "battery_voltage",

        "status",

        "estimated_status",

        "fuel_imputed",

        "current_imputed",

        "battery_imputed",

        "status_imputed",

        "imputation_confidence",

        "imputation_confidence_level"

    ]



    available = [

        c

        for c in sample_columns

        if c in result.columns

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

        "GENERATED"

    )

    print(

        OUTPUT_FILE

    )

    print("=" * 70)





if __name__ == "__main__":

    main()