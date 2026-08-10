from pathlib import Path

import numpy as np
import pandas as pd


TEST_PATH = Path(
    "data/processed/test_fuel_forecasting.csv"
)


def main():

    print("\n" + "=" * 70)
    print("BASELINE ERROR ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(
        TEST_PATH,
        low_memory=False,
    )

    df = df[
        df["fuel_level_l"].notna()
        &
        df["target_fuel_3h"].notna()
    ].copy()

    # --------------------------------------------------
    # Actual 3-hour fuel change
    # --------------------------------------------------

    df["fuel_change_3h"] = (
        df["target_fuel_3h"]
        - df["fuel_level_l"]
    )

    df["absolute_error"] = (
        df["fuel_change_3h"].abs()
    )

    # --------------------------------------------------
    # Overall statistics
    # --------------------------------------------------

    print("\n3-Hour Fuel Change Statistics")

    print(
        df["fuel_change_3h"]
        .describe()
        .round(3)
    )

    print("\nAbsolute Change Statistics")

    print(
        df["absolute_error"]
        .describe(
            percentiles=[
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
        .round(3)
    )

    # --------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------

    print("\nPercentage of predictions within:")

    thresholds = [
        1,
        2,
        5,
        10,
        20,
        50,
        100,
    ]

    for threshold in thresholds:

        percentage = (
            (
                df["absolute_error"]
                <= threshold
            ).mean()
            * 100
        )

        print(
            f"±{threshold:>3} L : "
            f"{percentage:6.2f}%"
        )

    # --------------------------------------------------
    # Generator analysis
    # --------------------------------------------------

    print(
        "\nBaseline Performance by Generator"
    )

    generator_stats = (
        df.groupby("generator_id")
        .agg(
            records=(
                "absolute_error",
                "count",
            ),
            mean_error=(
                "absolute_error",
                "mean",
            ),
            median_error=(
                "absolute_error",
                "median",
            ),
            max_error=(
                "absolute_error",
                "max",
            ),
            mean_change=(
                "fuel_change_3h",
                "mean",
            ),
        )
        .sort_values(
            "mean_error",
            ascending=False,
        )
    )

    print(
        generator_stats.round(3)
    )

    # --------------------------------------------------
    # Largest errors
    # --------------------------------------------------

    print(
        "\nLargest 20 Baseline Errors"
    )

    largest = (
        df[
            [
                "generator_id",
                "timestamp",
                "fuel_level_l",
                "target_fuel_3h",
                "fuel_change_3h",
                "absolute_error",
            ]
        ]
        .sort_values(
            "absolute_error",
            ascending=False,
        )
        .head(20)
    )

    print(largest.to_string(index=False))

    # --------------------------------------------------
    # Running status
    # --------------------------------------------------

    if "estimated_status" in df.columns:

        print(
            "\nBaseline Error by Estimated Status"
        )

        status_stats = (
            df.groupby("estimated_status")
            .agg(
                records=(
                    "absolute_error",
                    "count",
                ),
                mean_error=(
                    "absolute_error",
                    "mean",
                ),
                median_error=(
                    "absolute_error",
                    "median",
                ),
            )
            .sort_values(
                "mean_error",
                ascending=False,
            )
        )

        print(
            status_stats.round(3)
        )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE ERROR ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()