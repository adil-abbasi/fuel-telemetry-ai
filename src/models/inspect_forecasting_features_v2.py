from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path(
    "data/processed/fuel_forecasting_features_v2.csv"
)


def main():

    print("\n" + "=" * 70)
    print("FORECASTING FEATURES V2 — LEAKAGE & QUALITY INSPECTION")
    print("=" * 70)

    print("\nLoading dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    print(
        f"Rows:    {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # ==================================================
    # BASIC DATA CHECK
    # ==================================================

    print("\n" + "=" * 70)
    print("1. BASIC DATA QUALITY")
    print("=" * 70)

    print(
        "\nDuplicate rows:",
        int(df.duplicated().sum()),
    )

    print(
        "Missing timestamps:",
        int(df["timestamp"].isna().sum()),
    )

    print(
        "Missing generator IDs:",
        int(df["generator_id"].isna().sum()),
    )

    print(
        "Generators:",
        df["generator_id"].nunique(),
    )

    # ==================================================
    # TARGET CHECK
    # ==================================================

    target = "target_fuel_3h"

    print("\n" + "=" * 70)
    print("2. TARGET CHECK")
    print("=" * 70)

    print(
        "\nTarget missing:",
        int(df[target].isna().sum()),
    )

    print(
        "Target negative:",
        int((df[target] < 0).sum()),
    )

    print(
        "\nTarget statistics:"
    )

    print(
        df[target].describe().round(3)
    )

    # ==================================================
    # POTENTIAL LEAKAGE
    # ==================================================

    print("\n" + "=" * 70)
    print("3. POTENTIAL TARGET LEAKAGE")
    print("=" * 70)

    leakage_keywords = [
        "target",
        "future",
        "forecast",
        "lookup",
    ]

    suspicious = []

    for column in df.columns:

        column_lower = column.lower()

        if any(
            keyword in column_lower
            for keyword in leakage_keywords
        ):

            if column != target:

                suspicious.append(column)

    if suspicious:

        print(
            "\nSuspicious columns:"
        )

        for column in suspicious:

            print(
                f"  [CHECK] {column}"
            )

    else:

        print(
            "\nNo suspicious columns found."
        )

    # ==================================================
    # EXPLICIT MODEL EXCLUSION LIST
    # ==================================================

    print("\n" + "=" * 70)
    print("4. MODEL EXCLUSION CHECK")
    print("=" * 70)

    excluded_columns = {
        "timestamp",
        "generator_id",

        # Target
        "target_fuel_3h",

        # Target construction metadata
        "target_lookup_time",
        "target_timestamp",
        "target_time_difference_minutes",
        "target_available",

        # Raw rate known to contain extreme values
        "fuel_rate_lph",
        "fuel_rate_lps",

        # Text / categorical fields for now
        "site_name",
        "estimated_status",
        "confidence_level",
        "imputation_confidence_level",
        "status",
        "validation_reason",
        "estimation_reason",

    }

    available_exclusions = [
        column
        for column in excluded_columns
        if column in df.columns
    ]

    print(
        "\nExcluded from model:"
    )

    for column in sorted(
        available_exclusions
    ):

        print(
            f"  - {column}"
        )

    # ==================================================
    # MODEL FEATURES
    # ==================================================

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    print(
        f"\nCandidate model features: "
        f"{len(feature_columns)}"
    )

    # ==================================================
    # NON-NUMERIC FEATURES
    # ==================================================

    print("\n" + "=" * 70)
    print("5. NON-NUMERIC FEATURE CHECK")
    print("=" * 70)

    non_numeric = []

    for column in feature_columns:

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):

            non_numeric.append(
                (
                    column,
                    str(df[column].dtype),
                )
            )

    if non_numeric:

        print(
            "\nNon-numeric candidate features:"
        )

        for column, dtype in non_numeric:

            print(
                f"  [CHECK] {column} ({dtype})"
            )

    else:

        print(
            "\nAll candidate features are numeric."
        )

    # ==================================================
    # MISSING VALUES
    # ==================================================

    print("\n" + "=" * 70)
    print("6. FEATURE MISSING VALUES")
    print("=" * 70)

    missing = (
        df[feature_columns]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]

    if len(missing):

        print(
            "\nFeatures with missing values:"
        )

        print(
            missing.head(30)
        )

    else:

        print(
            "\nNo missing feature values."
        )

    # ==================================================
    # INFINITE VALUES
    # ==================================================

    print("\n" + "=" * 70)
    print("7. INFINITE VALUES")
    print("=" * 70)

    numeric_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    infinite_counts = {}

    for column in numeric_features:

        count = np.isinf(
            df[column].to_numpy(
                dtype=float
            )
        ).sum()

        if count > 0:

            infinite_counts[
                column
            ] = int(count)

    if infinite_counts:

        for column, count in (
            infinite_counts.items()
        ):

            print(
                f"  [BAD] {column}: {count:,}"
            )

    else:

        print(
            "\nNo infinite values found."
        )

    # ==================================================
    # FEATURE CORRELATION
    # ==================================================

    print("\n" + "=" * 70)
    print("8. FEATURE / TARGET CORRELATION")
    print("=" * 70)

    correlation_features = [
        column
        for column in numeric_features
        if column != target
    ]

    correlation_data = (
        df[
            correlation_features
            + [target]
        ]
        .corr(numeric_only=True)[target]
        .drop(target)
        .abs()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nTop 20 absolute correlations:"
    )

    print(
        correlation_data
        .head(20)
        .round(4)
    )

    # ==================================================
    # GENERATOR COVERAGE
    # ==================================================

    print("\n" + "=" * 70)
    print("9. GENERATOR COVERAGE")
    print("=" * 70)

    coverage = (
        df.groupby(
            "generator_id"
        )
        .agg(
            records=(
                "generator_id",
                "size",
            ),
            valid_target=(
                target,
                lambda x:
                x.notna().sum(),
            ),
            start_time=(
                "timestamp",
                "min",
            ),
            end_time=(
                "timestamp",
                "max",
            ),
        )
    )

    coverage[
        "target_availability_pct"
    ] = (
        coverage["valid_target"]
        / coverage["records"]
        * 100
    )

    print(
        coverage
        .round(2)
        .to_string()
    )

    # ==================================================
    # ROLLING FEATURE CHECK
    # ==================================================

    print("\n" + "=" * 70)
    print("10. ROLLING FEATURE CHECK")
    print("=" * 70)

    rolling_features = [
        column
        for column in df.columns
        if any(
            keyword in column
            for keyword in [
                "_mean_5min",
                "_mean_15min",
                "_mean_30min",
                "_mean_60min",
                "_std_5min",
                "_std_15min",
                "_std_30min",
                "_std_60min",
            ]
        )
    ]

    print(
        f"\nRolling features found: "
        f"{len(rolling_features)}"
    )

    if rolling_features:

        print(
            "\nRolling feature missing percentages:"
        )

        rolling_missing = (
            df[rolling_features]
            .isna()
            .mean()
            * 100
        )

        print(
            rolling_missing
            .sort_values(
                ascending=False
            )
            .head(20)
            .round(2)
        )

    # ==================================================
    # FEATURE VARIANCE
    # ==================================================

    print("\n" + "=" * 70)
    print("11. ZERO-VARIANCE FEATURES")
    print("=" * 70)

    zero_variance = []

    for column in numeric_features:

        if (
            df[column].nunique(
                dropna=True
            )
            <= 1
        ):

            zero_variance.append(
                column
            )

    if zero_variance:

        for column in zero_variance:

            print(
                f"  [REMOVE] {column}"
            )

    else:

        print(
            "\nNo zero-variance features."
        )

    # ==================================================
    # FINAL SUMMARY
    # ==================================================

    print("\n" + "=" * 70)
    print("INSPECTION SUMMARY")
    print("=" * 70)

    print(
        f"\nTotal rows: "
        f"{len(df):,}"
    )

    print(
        f"Total columns: "
        f"{len(df.columns)}"
    )

    print(
        f"Candidate model features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Numeric model features: "
        f"{len(numeric_features)}"
    )

    print(
        f"Suspicious leakage columns: "
        f"{len(suspicious)}"
    )

    print(
        f"Features with missing values: "
        f"{len(missing)}"
    )

    print(
        f"Features with infinity: "
        f"{len(infinite_counts)}"
    )

    print(
        f"Zero-variance features: "
        f"{len(zero_variance)}"
    )

    print("\n" + "=" * 70)
    print("V2 INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()