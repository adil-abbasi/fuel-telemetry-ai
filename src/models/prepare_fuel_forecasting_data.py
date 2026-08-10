from pathlib import Path

import pandas as pd


# ======================================================
# PATHS
# ======================================================

INPUT_PATH = Path(
    "data/processed/fuel_forecasting_dataset.csv"
)

OUTPUT_DIR = Path(
    "data/processed"
)


TRAIN_PATH = OUTPUT_DIR / (
    "train_fuel_forecasting.csv"
)

VALIDATION_PATH = OUTPUT_DIR / (
    "validation_fuel_forecasting.csv"
)

TEST_PATH = OUTPUT_DIR / (
    "test_fuel_forecasting.csv"
)


# ======================================================
# SPLIT CONFIGURATION
# ======================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ======================================================
# FEATURES
# ======================================================

FEATURE_COLUMNS = [
    "fuel_level_l",
    "fuel_rate_lph",
    "fuel_delta",
    "current",
    "current_delta",
    "battery_voltage",
    "voltage_delta",
    "running_probability",
    "hour",
    "minute",
    "weekday",
    "is_weekend",
    "time_delta_sec",
    "telemetry_quality_score",
]


TARGET_COLUMN = "target_fuel_3h"


# ======================================================
# MAIN
# ======================================================

def main():

    print("\n" + "=" * 70)
    print("PREPARING FUEL FORECASTING DATA")
    print("=" * 70)

    # ==================================================
    # LOAD DATA
    # ==================================================

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
        f"Original rows: {len(df):,}"
    )

    # ==================================================
    # SORT
    # ==================================================

    df = df.sort_values(
        [
            "generator_id",
            "timestamp",
        ]
    ).reset_index(drop=True)

    # ==================================================
    # CHECK REQUIRED COLUMNS
    # ==================================================

    required_columns = [
        "generator_id",
        "timestamp",
        TARGET_COLUMN,
    ] + FEATURE_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # ==================================================
    # VALID TARGET
    # ==================================================

    print("\nFiltering invalid targets...")

    initial_rows = len(df)

    df = df[
        df[TARGET_COLUMN].notna()
    ].copy()

    # Fuel cannot physically be negative.
    df = df[
        df[TARGET_COLUMN] >= 0
    ].copy()

    removed_target_rows = (
        initial_rows - len(df)
    )

    print(
        "Rows removed because target "
        f"was missing/invalid: "
        f"{removed_target_rows:,}"
    )

    # ==================================================
    # FEATURE VALIDATION
    # ==================================================

    print("\nChecking feature missing values...")

    feature_missing = (
        df[FEATURE_COLUMNS]
        .isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    print(feature_missing)

    # ==================================================
    # NUMERIC FEATURE CLEANING
    # ==================================================

    print("\nPreparing numeric features...")

    for column in FEATURE_COLUMNS:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ==================================================
    # PHYSICALLY INVALID FUEL
    # ==================================================

    # Keep the original dataset untouched.
    # Only model input is constrained here.

    df.loc[
        df["fuel_level_l"] < 0,
        "fuel_level_l",
    ] = pd.NA

    # ==================================================
    # EXTREME FUEL RATE HANDLING
    # ==================================================

    # fuel_rate_lph contains extreme values caused
    # by very small time intervals.
    #
    # We do not delete the observations.
    # Instead, values outside a physically reasonable
    # range are treated as unavailable for the model.

    MAX_REASONABLE_FUEL_RATE = 100.0

    invalid_rate = (
        df["fuel_rate_lph"].abs()
        > MAX_REASONABLE_FUEL_RATE
    )

    invalid_rate_count = int(
        invalid_rate.sum()
    )

    df.loc[
        invalid_rate,
        "fuel_rate_lph",
    ] = pd.NA

    print(
        "Extreme fuel-rate values replaced: "
        f"{invalid_rate_count:,}"
    )

    # ==================================================
    # FEATURE MISSINGNESS
    # ==================================================

    # We do not drop rows just because a feature
    # is missing. XGBoost can handle missing
    # numerical values natively.

    print(
        "\nRows retained for modeling: "
        f"{len(df):,}"
    )

    # ==================================================
    # CHRONOLOGICAL SPLIT
    # ==================================================

    print(
        "\nCreating chronological "
        "train/validation/test split..."
    )

    train_parts = []
    validation_parts = []
    test_parts = []

    for generator_id, generator_df in df.groupby(
        "generator_id",
        sort=False,
    ):

        generator_df = generator_df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        n = len(generator_df)

        train_end = int(
            n * TRAIN_RATIO
        )

        validation_end = int(
            n
            * (
                TRAIN_RATIO
                + VALIDATION_RATIO
            )
        )

        train_part = generator_df.iloc[
            :train_end
        ].copy()

        validation_part = generator_df.iloc[
            train_end:validation_end
        ].copy()

        test_part = generator_df.iloc[
            validation_end:
        ].copy()

        train_parts.append(
            train_part
        )

        validation_parts.append(
            validation_part
        )

        test_parts.append(
            test_part
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True,
    )

    validation_df = pd.concat(
        validation_parts,
        ignore_index=True,
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True,
    )

    # ==================================================
    # FINAL SORT
    # ==================================================

    train_df = train_df.sort_values(
        [
            "generator_id",
            "timestamp",
        ]
    ).reset_index(drop=True)

    validation_df = validation_df.sort_values(
        [
            "generator_id",
            "timestamp",
        ]
    ).reset_index(drop=True)

    test_df = test_df.sort_values(
        [
            "generator_id",
            "timestamp",
        ]
    ).reset_index(drop=True)

    # ==================================================
    # SAVE
    # ==================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        TRAIN_PATH,
        index=False,
    )

    validation_df.to_csv(
        VALIDATION_PATH,
        index=False,
    )

    test_df.to_csv(
        TEST_PATH,
        index=False,
    )

    # ==================================================
    # REPORT
    # ==================================================

    print("\n" + "=" * 70)
    print("DATASET SPLIT SUMMARY")
    print("=" * 70)

    print(
        f"\nTraining rows:     "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows:  "
        f"{len(validation_df):,}"
    )

    print(
        f"Testing rows:     "
        f"{len(test_df):,}"
    )

    print("\nGenerator counts")

    print(
        "Training:",
        train_df["generator_id"].nunique(),
    )

    print(
        "Validation:",
        validation_df["generator_id"].nunique(),
    )

    print(
        "Testing:",
        test_df["generator_id"].nunique(),
    )

    # ==================================================
    # TIME BOUNDARIES
    # ==================================================

    print("\nTime boundaries")

    print(
        "\nTraining:"
    )

    print(
        train_df["timestamp"].min(),
        "→",
        train_df["timestamp"].max(),
    )

    print(
        "\nValidation:"
    )

    print(
        validation_df["timestamp"].min(),
        "→",
        validation_df["timestamp"].max(),
    )

    print(
        "\nTesting:"
    )

    print(
        test_df["timestamp"].min(),
        "→",
        test_df["timestamp"].max(),
    )

    # ==================================================
    # TARGET DISTRIBUTION
    # ==================================================

    print("\nTarget statistics")

    print(
        "\nTraining:"
    )

    print(
        train_df[TARGET_COLUMN]
        .describe()
        .round(2)
    )

    print(
        "\nValidation:"
    )

    print(
        validation_df[TARGET_COLUMN]
        .describe()
        .round(2)
    )

    print(
        "\nTesting:"
    )

    print(
        test_df[TARGET_COLUMN]
        .describe()
        .round(2)
    )

    # ==================================================
    # OUTPUT
    # ==================================================

    print("\nGenerated files:")

    print(
        TRAIN_PATH
    )

    print(
        VALIDATION_PATH
    )

    print(
        TEST_PATH
    )

    print("\n" + "=" * 70)
    print("FORECASTING DATA PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()