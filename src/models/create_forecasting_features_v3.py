"""
V3 FORECASTING FEATURE ENGINEERING

Purpose:
    Create a leakage-safe feature dataset for 3-hour fuel forecasting.

Important:
    - Preserve target_fuel_3h exactly from the source dataset.
    - DO NOT convert negative targets to NaN.
    - Flag suspicious negative targets separately.
    - Create rolling features per generator in chronological order.
    - Do not use future target information as model features.
    - NEVER merge rolling results back on generator_id + timestamp because
      duplicate timestamps can create a many-to-many/cartesian explosion.
    - Rolling features are assigned back by original row position instead.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path("data/processed/fuel_forecasting_dataset.csv")
OUTPUT_FILE = Path("data/processed/fuel_forecasting_features_v3.csv")


# ============================================================
# CONFIGURATION
# ============================================================

ROLLING_WINDOWS = {
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "60min": "60min",
}

ROLLING_MIN_PERIODS = 3
EPSILON = 1e-6


# ============================================================
# HELPERS
# ============================================================

def safe_numeric(df, columns):
    """Convert existing columns to numeric safely."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_time_features(df):
    print("\nCreating time features...")

    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["weekday"] = df["timestamp"].dt.weekday
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month

    df["is_weekend"] = (df["weekday"] >= 5).astype("int8")

    # Cyclic time encoding helps tree/ML models represent periodicity.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)

    return df


def add_basic_sensor_features(df):
    print("\nCreating basic sensor features...")

    group = df.groupby("generator_id", sort=False)

    # --------------------------------------------------------
    # Fuel
    # --------------------------------------------------------

    df["fuel_delta"] = group["fuel_level_l"].diff()

    df["time_delta_sec"] = (
        group["timestamp"]
        .diff()
        .dt.total_seconds()
        .clip(lower=0)
    )

    valid_time = df["time_delta_sec"] > EPSILON

    df["fuel_rate_lps"] = np.where(
        valid_time,
        df["fuel_delta"] / df["time_delta_sec"],
        np.nan,
    )

    df["fuel_rate_lph"] = df["fuel_rate_lps"] * 3600.0

    df["fuel_rate_lps"] = (
        df["fuel_rate_lps"]
        .replace([np.inf, -np.inf], np.nan)
    )

    df["fuel_rate_lph"] = (
        df["fuel_rate_lph"]
        .replace([np.inf, -np.inf], np.nan)
    )

    # --------------------------------------------------------
    # Current / voltage deltas
    # --------------------------------------------------------

    df["current_delta"] = group["current"].diff()
    df["voltage_delta"] = group["battery_voltage"].diff()

    # --------------------------------------------------------
    # Missing/invalid sensor indicators
    # --------------------------------------------------------

    df["fuel_missing"] = df["fuel_level_l"].isna().astype("int8")
    df["current_missing"] = df["current"].isna().astype("int8")
    df["battery_missing"] = df["battery_voltage"].isna().astype("int8")

    df["fuel_invalid"] = (
        df["fuel_level_l"] < 0
    ).astype("int8")

    df["current_invalid"] = (
        df["current"] < 0
    ).astype("int8")

    df["battery_invalid"] = (
        df["battery_voltage"] < 0
    ).astype("int8")

    # --------------------------------------------------------
    # Outlier indicators
    # --------------------------------------------------------

    df["fuel_outlier"] = (
        df["fuel_delta"].abs() > 100
    ).astype("int8")

    df["current_outlier"] = (
        df["current_delta"].abs() > 100
    ).astype("int8")

    df["voltage_outlier"] = (
        df["voltage_delta"].abs() > 50
    ).astype("int8")

    # --------------------------------------------------------
    # Timestamp gaps
    # --------------------------------------------------------

    df["timestamp_gap_seconds"] = df["time_delta_sec"]

    df["timestamp_gap"] = (
        df["time_delta_sec"] > 180
    ).astype("int8")

    return df


def add_fuel_behavior_features(df):
    print("\nCreating fuel behavior features...")

    group = df.groupby("generator_id", sort=False)

    # These are row-based lag features, not future-looking features.
    # The telemetry is approximately one minute apart, so these provide
    # useful short/medium-term historical change signals.
    for periods in [15, 30, 60]:
        df[f"fuel_change_{periods}row"] = (
            group["fuel_level_l"].diff(periods)
        )

    # Keep the original feature names expected by downstream code.
    df["fuel_change_15min"] = df["fuel_change_15row"]
    df["fuel_change_30min"] = df["fuel_change_30row"]
    df["fuel_change_60min"] = df["fuel_change_60row"]

    df["fuel_refill_signal"] = (
        df["fuel_change_60min"] > 50
    ).astype("int8")

    df["fuel_drop_signal"] = (
        df["fuel_change_60min"] < -50
    ).astype("int8")

    return df


def _rolling_values_by_generator(df, column, window, min_periods):
    """
    Calculate time-based rolling mean/std independently for each generator
    and return arrays aligned to the dataframe's current row positions.

    This intentionally avoids merge().

    Why:
        A generator can contain duplicate timestamps. Merging rolling results
        on generator_id + timestamp can therefore turn N rows into N x N rows,
        causing huge memory allocation and corrupting the dataset.
    """

    mean_values = np.full(len(df), np.nan, dtype=np.float64)
    std_values = np.full(len(df), np.nan, dtype=np.float64)

    grouped = df.groupby("generator_id", sort=False, dropna=False)

    for _, group_df in grouped:
        positions = group_df.index.to_numpy()

        values = group_df[column]
        timestamps = group_df["timestamp"]

        # Rolling requires chronological order.
        order = np.argsort(
            timestamps.to_numpy(dtype="datetime64[ns]"),
            kind="mergesort",
        )

        sorted_positions = positions[order]
        sorted_values = values.to_numpy(dtype=float)[order]
        sorted_timestamps = timestamps.to_numpy(dtype="datetime64[ns]")[order]

        series = pd.Series(
            sorted_values,
            index=pd.DatetimeIndex(sorted_timestamps),
        )

        rolling = series.rolling(
            window=window,
            min_periods=min_periods,
        )

        mean_result = rolling.mean().to_numpy(dtype=float)
        std_result = rolling.std().to_numpy(dtype=float)

        mean_values[sorted_positions] = mean_result
        std_values[sorted_positions] = std_result

    return mean_values, std_values


def _rolling_mean_by_generator(df, column, window, min_periods):
    """Position-aligned time-based rolling mean without any merge."""

    mean_values = np.full(len(df), np.nan, dtype=np.float64)

    grouped = df.groupby("generator_id", sort=False, dropna=False)

    for _, group_df in grouped:
        positions = group_df.index.to_numpy()

        timestamps = group_df["timestamp"]
        values = group_df[column]

        order = np.argsort(
            timestamps.to_numpy(dtype="datetime64[ns]"),
            kind="mergesort",
        )

        sorted_positions = positions[order]
        sorted_values = values.to_numpy(dtype=float)[order]
        sorted_timestamps = timestamps.to_numpy(dtype="datetime64[ns]")[order]

        series = pd.Series(
            sorted_values,
            index=pd.DatetimeIndex(sorted_timestamps),
        )

        result = (
            series
            .rolling(
                window=window,
                min_periods=min_periods,
            )
            .mean()
            .to_numpy(dtype=float)
        )

        mean_values[sorted_positions] = result

    return mean_values


def add_rolling_features(df):
    print("\nCreating rolling fuel/current/battery features...")

    # Work with a fresh positional index so every output array has exactly
    # one value per source row.
    df = df.sort_values(
        ["generator_id", "timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)

    numeric_columns = {
        "fuel": "fuel_level_l",
        "current": "current",
        "battery": "battery_voltage",
    }

    for feature_name, column in numeric_columns.items():
        print(f"\n{feature_name}:")

        if column not in df.columns:
            continue

        for label, window in ROLLING_WINDOWS.items():
            print(label)

            mean_values, std_values = _rolling_values_by_generator(
                df,
                column,
                window,
                ROLLING_MIN_PERIODS,
            )

            df[f"{feature_name}_mean_{label}"] = mean_values
            df[f"{feature_name}_std_{label}"] = std_values

    return df


def add_running_probability_features(df):
    print("\nCreating operating-state features...")

    if "running_probability" not in df.columns:
        print("running_probability not found; skipping.")
        return df

    for minutes in [5, 15, 30, 60]:
        window = f"{minutes}min"

        values = _rolling_mean_by_generator(
            df,
            "running_probability",
            window,
            ROLLING_MIN_PERIODS,
        )

        df[f"running_probability_mean_{minutes}min"] = values

    return df


def add_quality_features(df):
    print("\nCreating telemetry quality features...")

    if "telemetry_quality_score" not in df.columns:
        print(
            "telemetry_quality_score not found; "
            "creating NaN quality feature."
        )
        df["telemetry_quality_score"] = np.nan

    for minutes in [5, 15, 30, 60]:
        window = f"{minutes}min"

        values = _rolling_mean_by_generator(
            df,
            "telemetry_quality_score",
            window,
            ROLLING_MIN_PERIODS,
        )

        df[f"telemetry_quality_mean_{minutes}min"] = values

    return df


def add_fuel_trend_features(df):
    print("\nCreating fuel trend features...")

    group = df.groupby("generator_id", sort=False)

    for periods in [15, 30, 60]:
        df[f"fuel_trend_{periods}min"] = (
            group["fuel_level_l"].diff(periods)
        )

    return df


def add_target_integrity_features(df):
    """
    Preserve target_fuel_3h exactly.

    Negative targets are NOT deleted or converted to NaN.
    They are flagged for later modeling decisions.
    """

    print("\nAnalyzing target integrity...")

    if "target_fuel_3h" not in df.columns:
        raise ValueError(
            "target_fuel_3h column is missing."
        )

    df["target_negative"] = (
        df["target_fuel_3h"] < 0
    ).astype("int8")

    df["target_negative_024"] = (
        np.isclose(
            df["target_fuel_3h"].to_numpy(dtype=float),
            -0.24,
            atol=EPSILON,
            equal_nan=False,
        )
    ).astype("int8")

    df["target_valid_for_regression"] = (
        df["target_fuel_3h"].notna()
        & (df["target_fuel_3h"] >= 0)
    ).astype("int8")

    print(
        "\nNegative targets:",
        int(df["target_negative"].sum()),
    )

    print(
        "Negative -0.24 targets:",
        int(df["target_negative_024"].sum()),
    )

    return df


def add_site_sensor_diagnostics(df):
    print("\nCreating sensor diagnostic features...")

    if "battery_voltage" in df.columns:
        df["battery_zero"] = (
            df["battery_voltage"] <= 0
        ).astype("int8")

    if "fuel_level_l" in df.columns:
        df["fuel_zero_or_negative"] = (
            df["fuel_level_l"] <= 0
        ).astype("int8")

    if "current" in df.columns:
        df["current_active"] = (
            df["current"] > 5
        ).astype("int8")

    if (
        "current_active" in df.columns
        and "battery_zero" in df.columns
    ):
        df["sensor_state_conflict"] = (
            (df["current_active"] == 1)
            & (df["battery_zero"] == 1)
        ).astype("int8")

    # General missingness count is useful for model confidence.
    sensor_missing_columns = [
        c for c in [
            "fuel_level_l",
            "current",
            "battery_voltage",
        ]
        if c in df.columns
    ]

    if sensor_missing_columns:
        df["sensor_missing_count"] = (
            df[sensor_missing_columns]
            .isna()
            .sum(axis=1)
            .astype("int8")
        )

    return df


def remove_duplicate_columns(df):
    """
    Defensive protection against duplicate column labels.
    """

    return df.loc[:, ~df.columns.duplicated()].copy()


def validate_output(df, source_df):
    print("\n" + "=" * 70)
    print("V3 DATASET VALIDATION")
    print("=" * 70)

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # --------------------------------------------------------
    # Row-count integrity
    # --------------------------------------------------------

    source_rows = len(source_df)

    print(f"Source rows: {source_rows:,}")

    if len(df) != source_rows:
        raise ValueError(
            "ROW COUNT CHANGED during feature engineering: "
            f"source={source_rows:,}, output={len(df):,}"
        )

    # --------------------------------------------------------
    # Target integrity
    # --------------------------------------------------------

    if not df["target_fuel_3h"].equals(
        source_df["target_fuel_3h"]
    ):
        raise ValueError(
            "TARGET INTEGRITY FAILURE: "
            "target_fuel_3h changed during feature engineering."
        )

    negative_count = int(
        (df["target_fuel_3h"] < 0).sum()
    )

    print(
        f"\nNegative targets preserved: {negative_count:,}"
    )

    print(
        "Target missing:",
        int(df["target_fuel_3h"].isna().sum()),
    )

    # --------------------------------------------------------
    # Generator/timestamp duplicate check
    # --------------------------------------------------------

    duplicate_count = int(
        df.duplicated(
            subset=["generator_id", "timestamp"]
        ).sum()
    )

    print(
        "Generator/timestamp duplicates:",
        duplicate_count,
    )

    if duplicate_count > 0:
        print(
            "WARNING: duplicate generator/timestamp rows exist "
            "in the source data. They were preserved and were NOT "
            "used as merge keys."
        )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    numeric = df.select_dtypes(include=[np.number])

    if not numeric.empty:
        infinity_count = int(
            np.isinf(
                numeric.to_numpy()
            ).sum()
        )
    else:
        infinity_count = 0

    print(
        "Infinite numeric values:",
        infinity_count,
    )

    if infinity_count != 0:
        raise ValueError(
            "Infinite numeric values remain in the output."
        )

    # --------------------------------------------------------
    # Duplicate column names
    # --------------------------------------------------------

    duplicate_columns = (
        df.columns[df.columns.duplicated()]
        .tolist()
    )

    if duplicate_columns:
        raise ValueError(
            "Duplicate column names remain: "
            + str(duplicate_columns)
        )

    print(
        "Duplicate column names: 0"
    )

    # --------------------------------------------------------
    # Suspicious target-derived columns
    # --------------------------------------------------------

    suspicious = [
        c for c in df.columns
        if any(
            keyword in c.lower()
            for keyword in [
                "target_timestamp",
                "target_lookup",
                "target_time_difference",
            ]
        )
    ]

    print(
        "\nSuspicious target-derived columns:"
    )

    if suspicious:
        for column in suspicious:
            print(f"  [CHECK] {column}")
    else:
        print("  None")

    # --------------------------------------------------------
    # Target availability
    # --------------------------------------------------------

    print(
        "\nTarget availability by generator:"
    )

    availability = (
        df.groupby("generator_id")["target_fuel_3h"]
        .agg(
            records="size",
            valid_targets=lambda x: x.notna().sum(),
            negative_targets=lambda x: (x < 0).sum(),
        )
    )

    availability["availability_pct"] = (
        availability["valid_targets"]
        / availability["records"]
        * 100
    )

    print(
        availability.round(2)
    )

    # --------------------------------------------------------
    # Target comparison summary
    # --------------------------------------------------------

    source_negative = int(
        (source_df["target_fuel_3h"] < 0).sum()
    )

    source_missing = int(
        source_df["target_fuel_3h"].isna().sum()
    )

    output_missing = int(
        df["target_fuel_3h"].isna().sum()
    )

    print("\nTarget integrity summary:")
    print(
        f"  Source negative targets: {source_negative:,}"
    )
    print(
        f"  Output negative targets: {negative_count:,}"
    )
    print(
        f"  Source missing targets:  {source_missing:,}"
    )
    print(
        f"  Output missing targets:  {output_missing:,}"
    )

    if source_negative != negative_count:
        raise ValueError(
            "Negative target count changed."
        )

    if source_missing != output_missing:
        raise ValueError(
            "Missing target count changed."
        )

    print("\nValidation PASSED.")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("CREATING V3 FUEL FORECASTING FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\nLoading dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"],
    )

    print(
        f"Rows: {len(df):,}"
    )

    # Keep a copy of the original target before ANY feature work.
    source_target = df["target_fuel_3h"].copy()

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required = [
        "timestamp",
        "generator_id",
        "fuel_level_l",
        "current",
        "battery_voltage",
        "target_fuel_3h",
    ]

    missing_required = [
        c for c in required
        if c not in df.columns
    ]

    if missing_required:
        raise ValueError(
            "Missing required columns: "
            + str(missing_required)
        )

    # --------------------------------------------------------
    # Timestamp validation
    # --------------------------------------------------------

    if df["timestamp"].isna().any():
        raise ValueError(
            "timestamp contains missing values."
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    df = safe_numeric(
        df,
        [
            "fuel_level_l",
            "current",
            "battery_voltage",
            "target_fuel_3h",
            "running_probability",
            "telemetry_quality_score",
        ],
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    print(
        "\nSorting chronologically per generator..."
    )

    df = df.sort_values(
        ["generator_id", "timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Preserve target after sorting
    # --------------------------------------------------------

    # Rebuild the source target in the exact sorted order so validation
    # compares the final target to the correct original row.
    source_for_validation = df[
        ["generator_id", "timestamp", "target_fuel_3h"]
    ].copy()

    # --------------------------------------------------------
    # Target integrity FIRST
    # --------------------------------------------------------

    df = add_target_integrity_features(df)

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    df = add_time_features(df)
    df = add_basic_sensor_features(df)
    df = add_fuel_behavior_features(df)
    df = add_rolling_features(df)
    df = add_running_probability_features(df)
    df = add_quality_features(df)
    df = add_fuel_trend_features(df)
    df = add_site_sensor_diagnostics(df)

    # --------------------------------------------------------
    # Clean infinite values in features only.
    #
    # target_fuel_3h is intentionally NOT modified.
    # --------------------------------------------------------

    print(
        "\nReplacing infinite feature values..."
    )

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    feature_numeric_columns = [
        c for c in numeric_columns
        if c != "target_fuel_3h"
    ]

    if feature_numeric_columns:
        df[feature_numeric_columns] = (
            df[feature_numeric_columns]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    # --------------------------------------------------------
    # Remove duplicate columns
    # --------------------------------------------------------

    df = remove_duplicate_columns(df)

    # --------------------------------------------------------
    # Final target check before validation
    # --------------------------------------------------------

    if not df["target_fuel_3h"].equals(
        source_for_validation["target_fuel_3h"]
    ):
        raise ValueError(
            "Target changed before final save."
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    validate_output(
        df,
        source_for_validation,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nSaving V3 dataset..."
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Post-save existence check
    # --------------------------------------------------------

    if not OUTPUT_FILE.exists():
        raise IOError(
            f"Output file was not created: {OUTPUT_FILE}"
        )

    print(
        "\n" + "=" * 70
    )
    print(
        "V3 FEATURE ENGINEERING COMPLETE"
    )
    print(
        "=" * 70
    )

    print(
        f"\nSaved:"
    )
    print(
        OUTPUT_FILE
    )

    print(
        f"\nFinal rows: {len(df):,}"
    )
    print(
        f"Final columns: {len(df.columns):,}"
    )


if __name__ == "__main__":
    main()
