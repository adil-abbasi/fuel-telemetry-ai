"""
==============================================================
Telemetry Quality Scorer Test
==============================================================

Runs telemetry quality analysis
for all generators.

==============================================================
"""


import pandas as pd

from src.preprocessing.telemetry_quality_scorer import (
    TelemetryQualityScorer
)

from src.preprocessing.reporting_interval import (
    ReportingIntervalDetector
)



# ==========================================================
# Configuration
# ==========================================================


DATA_PATH = "data/raw/telemetry.csv"



# ==========================================================
# Main
# ==========================================================


def main():


    print("\nLoading dataset...\n")


    df = pd.read_csv(
        DATA_PATH
    )


    print(
        "Preparing timestamps..."
    )


    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )



    # ------------------------------------------------------
    # Get reporting intervals
    # ------------------------------------------------------

    print(
        "\nDetecting reporting intervals..."
    )


    detector = ReportingIntervalDetector(
        df
    )


    detector.run()


    interval_dictionary = (
        detector.get_interval_dictionary()
    )


    print(
        "\nInterval Dictionary:"
    )

    print(
        interval_dictionary
    )



    # ------------------------------------------------------
    # Quality scoring
    # ------------------------------------------------------

    print(
        "\nRunning Telemetry Quality Scorer..."
    )


    scorer = TelemetryQualityScorer(

        df,

        interval_dictionary

    )


    result = scorer.run()



    print()

    print("="*70)

    print(
        "TELEMETRY QUALITY REPORT"
    )

    print("="*70)


    print(
        result
    )


    print()

    print(
        "Average Quality Score:"
    )

    print(
        round(
            result["quality_score"]
            .mean(),
            2
        )
    )



    print()

    print(
        "Health Distribution"
    )


    print(

        result[
            "communication_health"
        ]
        .value_counts()

    )



    print()

    print(
        "Saving output..."
    )


    result.to_csv(

        "telemetry_quality_report.csv",

        index=False

    )


    print(
        "Generated:"
    )

    print(
        "- telemetry_quality_report.csv"
    )



if __name__ == "__main__":

    main()