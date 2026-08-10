
"""
CREATE 3-HOUR FUEL FORECASTING TARGET

Purpose:
    Create a leakage-safe target for 3-hour-ahead fuel forecasting.

Target definition:
    For each telemetry row:

        target_lookup_time = timestamp + 3 hours

    Find the FIRST valid observation for the SAME generator at or AFTER
    target_lookup_time, provided it is no more than 5 minutes later.

Important:
    - NEVER use a past observation as the future target.
    - NEVER use direction="nearest".
    - The target is based on timestamp, not row count.
    - Duplicate generator/timestamp observations are preserved.
    - No future information is added to model features here.
"""

from pathlib import Path

import pandas as pd


# ======================================================
# PATHS
# ======================================================

INPUT_PATH = Path(
    "data/processed/imputed_telemetry_dataset.csv"
)

OUTPUT_PATH = Path(
    "data/processed/fuel_forecasting_dataset.csv"
)


# ======================================================
# CONFIGURATION
# ======================================================

FORECAST_HOURS = 3

# Maximum acceptable delay AFTER the exact 3-hour
# lookup timestamp.
TARGET_TOLERANCE_MINUTES = 5


# ======================================================
# MAIN
# ======================================================

def main():

    print("\n" + "=" * 70)
    print("CREATING 3-HOUR FUEL FORECASTING TARGET")
    print("=" * 70)

    # ==================================================
    # LOAD
    # ==================================================

    print("\nLoading dataset...")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found:\n{INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    print(
        f"Input rows: {len(df):,}"
    )

    # ==================================================
    # TIMESTAMP
    # ==================================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        invalid_timestamps = int(
            df["timestamp"].isna().sum()
        )

        raise ValueError(
            f"Found {invalid_timestamps:,} invalid timestamps."
        )

    # ==================================================
    # REQUIRED COLUMNS
    # ==================================================

    required_columns = [
        "generator_id",
        "timestamp",
        "fuel_level_l",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    # ==================================================
    # NUMERIC FUEL
    # ==================================================

    df["fuel_level_l"] = pd.to_numeric(
        df["fuel_level_l"],
        errors="coerce",
    )

    # ==================================================
    # SORT SOURCE
    # ==================================================

    df = df.sort_values(
        [
            "generator_id",
            "timestamp",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # ==================================================
    # PRESERVE ORIGINAL ROW ID
    # ==================================================

    # This allows us to restore the exact original
    # row ordering after merge_asof.

    df["_source_row_id"] = range(len(df))

    # ==================================================
    # CREATE TARGET LOOKUP TIME
    # ==================================================

    df["target_lookup_time"] = (
        df["timestamp"]
        + pd.Timedelta(
            hours=FORECAST_HOURS
        )
    )

    # ==================================================
    # FUTURE TARGET TABLE
    # ==================================================

    future = df[
        [
            "generator_id",
            "timestamp",
            "fuel_level_l",
        ]
    ].copy()

    future = future.rename(
        columns={
            "timestamp": "target_timestamp",
            "fuel_level_l": "target_fuel_3h",
        }
    )

    # --------------------------------------------------
    # Only observations with an actual fuel value can
    # be used as the target.
    # --------------------------------------------------

    future = future[
        future["target_fuel_3h"].notna()
    ].copy()

    # --------------------------------------------------
    # Negative fuel observations are not valid physical
    # target observations.
    #
    # We do NOT modify the original dataframe.
    # --------------------------------------------------

    future = future[
        future["target_fuel_3h"] >= 0
    ].copy()

    # ==================================================
    # SORT FOR merge_asof
    # ==================================================

    # IMPORTANT:
    #
    # merge_asof requires the time key to be globally
    # sorted.
    #
    # We therefore sort by timestamp first and generator
    # second.

    source_lookup = df[
        [
            "_source_row_id",
            "generator_id",
            "timestamp",
            "target_lookup_time",
        ]
    ].copy()

    source_lookup = source_lookup.sort_values(
        [
            "target_lookup_time",
            "generator_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    future = future.sort_values(
        [
            "target_timestamp",
            "generator_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # ==================================================
    # FUTURE-ONLY AS-OF LOOKUP
    # ==================================================

    print(
        "\nSearching for future observations..."
    )

    result = pd.merge_asof(
        source_lookup,
        future,
        left_on="target_lookup_time",
        right_on="target_timestamp",
        by="generator_id",

        # ------------------------------------------------
        # CRITICAL:
        #
        # "forward" means the matched observation must be
        # AT or AFTER the desired 3-hour timestamp.
        #
        # This prevents accidental past matching.
        # ------------------------------------------------
        direction="forward",

        # Maximum allowed difference.
        tolerance=pd.Timedelta(
            minutes=TARGET_TOLERANCE_MINUTES
        ),
    )

    # ==================================================
    # CALCULATE TARGET TIMING
    # ==================================================

    result["target_time_difference_minutes"] = (
        (
            result["target_timestamp"]
            - result["target_lookup_time"]
        )
        .dt.total_seconds()
        / 60.0
    )

    # ==================================================
    # VALID TARGET
    # ==================================================

    result["target_available"] = (
        result["target_fuel_3h"].notna()
        &
        result["target_timestamp"].notna()
        &
        (
            result["target_time_difference_minutes"]
            >= 0
        )
        &
        (
            result["target_time_difference_minutes"]
            <= TARGET_TOLERANCE_MINUTES
        )
    )

    # ==================================================
    # SAFETY CHECK
    # ==================================================

    # This should NEVER happen with direction="forward".

    invalid_direction = (
        result["target_available"]
        &
        (
            result["target_time_difference_minutes"]
            < 0
        )
    )

    if invalid_direction.any():

        count = int(
            invalid_direction.sum()
        )

        raise ValueError(
            "TARGET LEAKAGE DETECTED: "
            f"{count:,} targets matched a past observation."
        )

    # ==================================================
    # RESTORE SOURCE ORDER
    # ==================================================

    result = result.sort_values(
        "_source_row_id"
    ).reset_index(drop=True)

    # ==================================================
    # ALIGN TARGET BACK TO ORIGINAL DATAFRAME
    # ==================================================

    if len(result) != len(df):
        raise ValueError(
            "Target merge changed row count: "
            f"source={len(df):,}, "
            f"result={len(result):,}"
        )

    df["target_fuel_3h"] = (
        result["target_fuel_3h"]
        .to_numpy()
    )

    df["target_timestamp"] = (
        result["target_timestamp"]
        .to_numpy()
    )

    df["target_time_difference_minutes"] = (
        result[
            "target_time_difference_minutes"
        ].to_numpy()
    )

    df["target_available"] = (
        result["target_available"]
        .to_numpy()
    )

    # ==================================================
    # TARGET STATISTICS
    # ==================================================

    total_rows = len(df)

    target_rows = int(
        df["target_available"].sum()
    )

    missing_target_rows = (
        total_rows - target_rows
    )

    target_percentage = (
        target_rows / total_rows * 100
        if total_rows > 0
        else 0
    )

    print("\n" + "=" * 70)
    print("TARGET STATISTICS")
    print("=" * 70)

    print(
        f"Total rows:              {total_rows:,}"
    )

    print(
        f"Valid 3-hour targets:    {target_rows:,}"
    )

    print(
        f"Missing 3-hour targets:  {missing_target_rows:,}"
    )

    print(
        f"Target availability:     {target_percentage:.2f}%"
    )

    # ==================================================
    # TARGET TIMING STATISTICS
    # ==================================================

    valid_differences = df.loc[
        df["target_available"],
        "target_time_difference_minutes",
    ]

    if len(valid_differences) > 0:

        print(
            "\nTarget timing difference "
            "(minutes AFTER desired +3h):"
        )

        print(
            valid_differences
            .describe()
            .round(3)
        )

        print(
            f"\nMaximum target delay: "
            f"{valid_differences.max():.3f} minutes"
        )

        print(
            f"Mean target delay: "
            f"{valid_differences.mean():.3f} minutes"
        )

    # ==================================================
    # TARGET FUEL STATISTICS
    # ==================================================

    print("\n" + "=" * 70)
    print("3-HOUR TARGET FUEL STATISTICS")
    print("=" * 70)

    valid_targets = df.loc[
        df["target_available"],
        "target_fuel_3h",
    ]

    if len(valid_targets) > 0:

        print(
            valid_targets
            .describe()
            .round(2)
        )

    # ==================================================
    # TARGET AVAILABILITY BY GENERATOR
    # ==================================================

    print("\n" + "=" * 70)
    print("TARGET AVAILABILITY BY GENERATOR")
    print("=" * 70)

    generator_stats = (
        df.groupby("generator_id")
        .agg(
            records=(
                "generator_id",
                "size",
            ),
            valid_targets=(
                "target_available",
                "sum",
            ),
        )
    )

    generator_stats["availability_percent"] = (
        generator_stats["valid_targets"]
        / generator_stats["records"]
        * 100
    )

    print(
        generator_stats.round(2)
    )

    # ==================================================
    # TARGET SANITY CHECKS
    # ==================================================

    print("\n" + "=" * 70)
    print("TARGET SANITY CHECKS")
    print("=" * 70)

    # --------------------------------------------------
    # No past target
    # --------------------------------------------------

    past_targets = (
        df["target_available"]
        &
        (
            df["target_timestamp"]
            < df["target_lookup_time"]
        )
    )

    print(
        f"Past target matches: "
        f"{int(past_targets.sum()):,}"
    )

    if past_targets.any():
        raise ValueError(
            "Past target matches detected."
        )

    # --------------------------------------------------
    # Target must be within tolerance
    # --------------------------------------------------

    excessive_delay = (
        df["target_available"]
        &
        (
            df["target_time_difference_minutes"]
            > TARGET_TOLERANCE_MINUTES
        )
    )

    print(
        f"Targets beyond tolerance: "
        f"{int(excessive_delay.sum()):,}"
    )

    if excessive_delay.any():
        raise ValueError(
            "Targets beyond allowed tolerance detected."
        )

    # --------------------------------------------------
    # Target timestamp must actually be later
    # --------------------------------------------------

    invalid_target_timestamp = (
        df["target_available"]
        &
        (
            df["target_timestamp"]
            < df["timestamp"]
        )
    )

    print(
        f"Targets before source timestamp: "
        f"{int(invalid_target_timestamp.sum()):,}"
    )

    if invalid_target_timestamp.any():
        raise ValueError(
            "Target timestamp occurs before source timestamp."
        )

    print(
        "\nTarget sanity checks PASSED."
    )

    # ==================================================
    # SAVE
    # ==================================================

    # Remove internal columns before saving.

    columns_to_drop = [
        "_source_row_id",
        "target_lookup_time",
    ]

    df = df.drop(
        columns=[
            column
            for column in columns_to_drop
            if column in df.columns
        ]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nSaving dataset..."
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ==================================================
    # FINAL CHECK
    # ==================================================

    if not OUTPUT_PATH.exists():
        raise IOError(
            f"Output file was not created:\n{OUTPUT_PATH}"
        )

    print("\n" + "=" * 70)
    print("TARGET CREATION COMPLETE")
    print("=" * 70)

    print(
        f"\nSaved:\n{OUTPUT_PATH}"
    )

    print(
        f"\nFinal rows: {len(df):,}"
    )

    print(
        f"Valid targets: "
        f"{int(df['target_available'].sum()):,}"
    )


if __name__ == "__main__":
    main()

