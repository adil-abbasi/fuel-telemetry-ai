from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PREDICTIONS_PATH = Path(
    "data/processed/forecasting/results/"
    "xgboost_baseline_test_predictions.csv"
)

OUTPUT_DIR = Path(
    "data/processed/forecasting/results"
)

ERROR_ANALYSIS_PATH = (
    OUTPUT_DIR / "xgboost_baseline_error_analysis.csv"
)


def header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:

    header("XGBOOST BASELINE ERROR ANALYSIS")

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    print(f"Rows loaded: {len(df):,}")

    required = [
        "target_fuel_3h",
        "prediction_fuel_3h",
        "prediction_error_l",
        "absolute_error_l",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # BASIC ERROR STATISTICS
    # --------------------------------------------------------

    header("OVERALL ERROR DISTRIBUTION")

    errors = df["prediction_error_l"]
    absolute_errors = df["absolute_error_l"]

    print(
        f"Mean error:       {errors.mean():.4f} L"
    )

    print(
        f"Median error:     {errors.median():.4f} L"
    )

    print(
        f"Mean absolute:    {absolute_errors.mean():.4f} L"
    )

    print(
        f"Median absolute:  {absolute_errors.median():.4f} L"
    )

    print(
        f"Std error:        {errors.std():.4f} L"
    )

    print(
        f"Minimum error:    {errors.min():.4f} L"
    )

    print(
        f"Maximum error:    {errors.max():.4f} L"
    )

    # --------------------------------------------------------
    # ERROR PERCENTILES
    # --------------------------------------------------------

    header("ABSOLUTE ERROR PERCENTILES")

    percentiles = [
        50,
        75,
        90,
        95,
        99,
        99.5,
        99.9,
    ]

    for percentile in percentiles:

        value = np.percentile(
            absolute_errors,
            percentile,
        )

        print(
            f"P{percentile:<5}: {value:.4f} L"
        )

    # --------------------------------------------------------
    # ERROR THRESHOLDS
    # --------------------------------------------------------

    header("ERROR THRESHOLDS")

    thresholds = [
        5,
        10,
        20,
        30,
        50,
        100,
        200,
    ]

    total = len(df)

    for threshold in thresholds:

        count = (
            absolute_errors > threshold
        ).sum()

        percentage = (
            count / total * 100
        )

        print(
            f"> {threshold:>3} L: "
            f"{count:>6,} rows "
            f"({percentage:>6.2f}%)"
        )

    # --------------------------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------------------------

    header("TARGET DISTRIBUTION")

    target = df["target_fuel_3h"]

    print(
        f"Minimum target:   {target.min():.4f} L"
    )

    print(
        f"Maximum target:   {target.max():.4f} L"
    )

    print(
        f"Mean target:      {target.mean():.4f} L"
    )

    print(
        f"Median target:    {target.median():.4f} L"
    )

    print(
        f"Std target:       {target.std():.4f} L"
    )

    # --------------------------------------------------------
    # WORST PREDICTIONS
    # --------------------------------------------------------

    header("TOP 30 WORST PREDICTIONS")

    worst = (
        df.sort_values(
            "absolute_error_l",
            ascending=False,
        )
        .head(30)
    )

    columns = [
        col
        for col in [
            "generator_id",
            "site_name",
            "timestamp",
            "fuel_level_l",
            "target_fuel_3h",
            "prediction_fuel_3h",
            "prediction_error_l",
            "absolute_error_l",
        ]
        if col in worst.columns
    ]

    print(
        worst[columns].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # GENERATOR-LEVEL ANALYSIS
    # --------------------------------------------------------

    if "generator_id" in df.columns:

        header("GENERATOR-LEVEL ERROR")

        generator_analysis = (
            df.groupby("generator_id")
            .agg(
                rows=(
                    "target_fuel_3h",
                    "size",
                ),
                mae=(
                    "absolute_error_l",
                    "mean",
                ),
                median_absolute_error=(
                    "absolute_error_l",
                    "median",
                ),
                rmse=(
                    "prediction_error_l",
                    lambda x: np.sqrt(
                        np.mean(x ** 2)
                    ),
                ),
                max_error=(
                    "absolute_error_l",
                    "max",
                ),
                mean_target=(
                    "target_fuel_3h",
                    "mean",
                ),
            )
            .sort_values(
                "mae",
                ascending=False,
            )
        )

        print(
            generator_analysis.to_string()
        )

        generator_path = (
            OUTPUT_DIR
            / "xgboost_baseline_generator_errors.csv"
        )

        generator_analysis.to_csv(
            generator_path
        )

        print(
            f"\nSaved:\n{generator_path}"
        )

    # --------------------------------------------------------
    # BIAS ANALYSIS
    # --------------------------------------------------------

    header("PREDICTION BIAS")

    overprediction = (
        errors > 0
    ).sum()

    underprediction = (
        errors < 0
    ).sum()

    exact = (
        errors == 0
    ).sum()

    print(
        f"Overprediction:   "
        f"{overprediction:,} "
        f"({overprediction / total * 100:.2f}%)"
    )

    print(
        f"Underprediction:  "
        f"{underprediction:,} "
        f"({underprediction / total * 100:.2f}%)"
    )

    print(
        f"Exact prediction: "
        f"{exact:,}"
    )

    # --------------------------------------------------------
    # LARGE ERROR BIAS
    # --------------------------------------------------------

    header("LARGE ERROR BIAS")

    large_error = df[
        df["absolute_error_l"] > 50
    ]

    print(
        f"Rows with >50 L error: "
        f"{len(large_error):,}"
    )

    if len(large_error) > 0:

        print(
            f"Mean error (>50 L): "
            f"{large_error['prediction_error_l'].mean():.4f} L"
        )

        print(
            f"Median error (>50 L): "
            f"{large_error['prediction_error_l'].median():.4f} L"
        )

        print(
            f"Mean target (>50 L): "
            f"{large_error['target_fuel_3h'].mean():.4f} L"
        )

        print(
            f"Mean prediction (>50 L): "
            f"{large_error['prediction_fuel_3h'].mean():.4f} L"
        )

    # --------------------------------------------------------
    # ERROR VS TARGET SIZE
    # --------------------------------------------------------

    header("ERROR BY TARGET RANGE")

    bins = [
        -np.inf,
        10,
        25,
        50,
        100,
        200,
        500,
        np.inf,
    ]

    labels = [
        "<=10",
        "10-25",
        "25-50",
        "50-100",
        "100-200",
        "200-500",
        ">500",
    ]

    df["target_range"] = pd.cut(
        df["target_fuel_3h"],
        bins=bins,
        labels=labels,
    )

    target_range_analysis = (
        df.groupby(
            "target_range",
            observed=False,
        )
        .agg(
            rows=(
                "target_fuel_3h",
                "size",
            ),
            mae=(
                "absolute_error_l",
                "mean",
            ),
            rmse=(
                "prediction_error_l",
                lambda x: np.sqrt(
                    np.mean(x ** 2)
                ),
            ),
            mean_target=(
                "target_fuel_3h",
                "mean",
            ),
        )
    )

    print(
        target_range_analysis.to_string()
    )

    # --------------------------------------------------------
    # SAVE COMPLETE ERROR ANALYSIS
    # --------------------------------------------------------

    df.to_csv(
        ERROR_ANALYSIS_PATH,
        index=False,
    )

    print()
    print(
        f"Error analysis saved:\n"
        f"{ERROR_ANALYSIS_PATH}"
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    header("ERROR ANALYSIS COMPLETE")

    print(
        "Next stage:"
    )

    print(
        "  Inspect worst predictions."
    )

    print(
        "  Inspect generator-level performance."
    )

    print(
        "  Identify distribution shift/outlier behavior."
    )

    print(
        "  Then tune XGBoost using validation data."
    )


if __name__ == "__main__":
    main()