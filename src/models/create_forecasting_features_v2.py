from pathlib import Path

import numpy as np
import pandas as pd


# ======================================================
# PATHS
# ======================================================

INPUT_PATH = Path(
    "data/processed/fuel_forecasting_dataset.csv"
)

OUTPUT_PATH = Path(
    "data/processed/fuel_forecasting_features_v2.csv"
)


# ======================================================
# CONFIGURATION
# ======================================================

ROLLING_WINDOWS = {
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "60min": "60min",
}


# ======================================================
# TIME-BASED ROLLING FEATURE
# ======================================================

def add_time_rolling_features(
    df,
    column,
    prefix,
    windows,
):
    """
    Creates genuine time-based rolling features
    independently for each generator.

    Only historical/current observations are used.
    """

    result = {}

    for name, window in windows.items():

        print(
            f"  {prefix}: {window}"
        )

        rolled_mean = (
            df.set_index("timestamp")
            .groupby("generator_id")[column]
            .rolling(
                window=window,
                min_periods=3,
            )
            .mean()
            .reset_index(
                level=[0, 1],
                drop=True,
            )
        )

        rolled_std = (
            df.set_index("timestamp")
            .groupby("generator_id")[column]
            .rolling(
                window=window,
                min_periods=3,
            )
            .std()
            .reset_index(
                level=[0, 1],
                drop=True,
            )
        )

        # The rolling result is indexed by timestamp.
        # Rebuild using a stable row key to preserve
        # exact alignment with the original dataframe.

        temp = df[
            [
                "generator_id",
                "timestamp",
            ]
        ].copy()

        temp["_row_id"] = np.arange(
            len(temp)
        )

        mean_temp = (
            temp.set_index(
                [
                    "generator_id",
                    "timestamp",
                ]
            )
            .assign(
                value=rolled_mean.values
            )
            .reset_index()
            .sort_values(
                "_row_id"
            )
        )

        std_temp = (
            temp.set_index(
                [
                    "generator_id",
                    "timestamp",
                ]
            )
            .assign(
                value=rolled_std.values
            )
            .reset_index()
            .sort_values(
                "_row_id"
            )
        )

        result[
            f"{prefix}_mean_{name}"
        ] = mean_temp[
            "value"
        ].to_numpy()

        result[
            f"{prefix}_std_{name}"
        ] = std_temp[
            "value"
        ].to_numpy()

    return result


# ======================================================
# SIMPLE GROUPED ROLLING
# ======================================================

def grouped_shift(
    df,
    column,
    periods,
):
    return (
        df.groupby(
            "generator_id"
        )[column]
        .shift(periods)
    )


# ======================================================
# MAIN
# ======================================================

def main():

    print("\n" + "=" * 70)
    print("CREATING FORECASTING FEATURES V2")
    print("=" * 70)

    # ==================================================
    # LOAD
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

    df = (
        df.sort_values(
            [
                "generator_id",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    # Stable row identifier.
    df["_row_id"] = np.arange(
        len(df)
    )

    print(
        f"Rows: {len(df):,}"
    )

    # ==================================================
    # NUMERIC CONVERSION
    # ==================================================

    numeric_columns = [
        "fuel_level_l",
        "fuel_delta",
        "current",
        "battery_voltage",
        "running_probability",
        "telemetry_quality_score",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # ==================================================
    # FUEL CHANGE
    # ==================================================

    print(
        "\nCreating fuel behavior features..."
    )

    df["fuel_change_15min"] = (
        df["fuel_level_l"]
        -
        grouped_shift(
            df,
            "fuel_level_l",
            15,
        )
    )

    df["fuel_change_30min"] = (
        df["fuel_level_l"]
        -
        grouped_shift(
            df,
            "fuel_level_l",
            30,
        )
    )

    df["fuel_change_60min"] = (
        df["fuel_level_l"]
        -
        grouped_shift(
            df,
            "fuel_level_l",
            60,
        )
    )

    # ==================================================
    # REFILL / DROP SIGNAL
    # ==================================================

    df["fuel_refill_signal"] = (
        df["fuel_change_15min"]
        > 20
    ).astype(int)

    df["fuel_drop_signal"] = (
        df["fuel_change_15min"]
        < -20
    ).astype(int)

    # ==================================================
    # TIME-BASED FUEL FEATURES
    # ==================================================

    print(
        "\nCreating rolling fuel features..."
    )

    fuel_features = (
        add_time_rolling_features(
            df,
            "fuel_level_l",
            "fuel",
            ROLLING_WINDOWS,
        )
    )

    for name, values in fuel_features.items():

        df[name] = values

    # ==================================================
    # CURRENT FEATURES
    # ==================================================

    if "current" in df.columns:

        print(
            "\nCreating rolling current features..."
        )

        current_features = (
            add_time_rolling_features(
                df,
                "current",
                "current",
                ROLLING_WINDOWS,
            )
        )

        for name, values in current_features.items():

            df[name] = values

    # ==================================================
    # BATTERY FEATURES
    # ==================================================

    if "battery_voltage" in df.columns:

        print(
            "\nCreating rolling battery features..."
        )

        battery_features = (
            add_time_rolling_features(
                df,
                "battery_voltage",
                "battery",
                ROLLING_WINDOWS,
            )
        )

        for name, values in battery_features.items():

            df[name] = values

    # ==================================================
    # RUNNING PROBABILITY
    # ==================================================

    print(
        "\nCreating operating-state features..."
    )

    for name, window in ROLLING_WINDOWS.items():

        running = (
            df.set_index(
                "timestamp"
            )
            .groupby(
                "generator_id"
            )["running_probability"]
            .rolling(
                window=window,
                min_periods=3,
            )
            .mean()
            .reset_index(
                level=[0, 1],
                drop=True,
            )
        )

        # Align through generator + timestamp.
        lookup = pd.DataFrame(
            {
                "generator_id":
                    df["generator_id"].values,
                "timestamp":
                    df["timestamp"].values,
                "value":
                    running.values,
            }
        )

        lookup = (
            lookup
            .sort_values(
                [
                    "generator_id",
                    "timestamp",
                ]
            )
            .reset_index(drop=True)
        )

        original_order = (
            df[
                [
                    "_row_id",
                    "generator_id",
                    "timestamp",
                ]
            ]
            .sort_values(
                [
                    "generator_id",
                    "timestamp",
                ]
            )
        )

        lookup["_row_id"] = (
            original_order["_row_id"]
            .values
        )

        lookup = lookup.sort_values(
            "_row_id"
        )

        df[
            f"running_probability_mean_{name}"
        ] = lookup[
            "value"
        ].values

    # ==================================================
    # TELEMETRY QUALITY
    # ==================================================

    print(
        "\nCreating telemetry quality features..."
    )

    for name, window in ROLLING_WINDOWS.items():

        quality = (
            df.set_index(
                "timestamp"
            )
            .groupby(
                "generator_id"
            )[
                "telemetry_quality_score"
            ]
            .rolling(
                window=window,
                min_periods=3,
            )
            .mean()
            .reset_index(
                level=[0, 1],
                drop=True,
            )
        )

        lookup = pd.DataFrame(
            {
                "generator_id":
                    df["generator_id"].values,
                "timestamp":
                    df["timestamp"].values,
                "value":
                    quality.values,
            }
        )

        lookup = (
            lookup
            .sort_values(
                [
                    "generator_id",
                    "timestamp",
                ]
            )
            .reset_index(drop=True)
        )

        original_order = (
            df[
                [
                    "_row_id",
                    "generator_id",
                    "timestamp",
                ]
            ]
            .sort_values(
                [
                    "generator_id",
                    "timestamp",
                ]
            )
        )

        lookup["_row_id"] = (
            original_order["_row_id"]
            .values
        )

        lookup = lookup.sort_values(
            "_row_id"
        )

        df[
            f"telemetry_quality_mean_{name}"
        ] = lookup[
            "value"
        ].values

    # ==================================================
    # FUEL TREND
    # ==================================================

    print(
        "\nCreating fuel trend features..."
    )

    df["fuel_trend_15min"] = (
        df["fuel_change_15min"]
        / 15.0
    )

    df["fuel_trend_30min"] = (
        df["fuel_change_30min"]
        / 30.0
    )

    df["fuel_trend_60min"] = (
        df["fuel_change_60min"]
        / 60.0
    )

    # ==================================================
    # EXTREME VALUES
    # ==================================================

    print(
        "\nHandling unrealistic fuel values..."
    )

    df.loc[
        df["fuel_level_l"] < 0,
        "fuel_level_l",
    ] = np.nan

    trend_columns = [
        "fuel_change_15min",
        "fuel_change_30min",
        "fuel_change_60min",
        "fuel_trend_15min",
        "fuel_trend_30min",
        "fuel_trend_60min",
    ]

    for column in trend_columns:

        df.loc[
            df[column].abs() > 500,
            column,
        ] = np.nan

    # ==================================================
    # TARGET
    # ==================================================

    if "target_fuel_3h" in df.columns:

        df["target_fuel_3h"] = pd.to_numeric(
            df["target_fuel_3h"],
            errors="coerce",
        )

        df.loc[
            df["target_fuel_3h"] < 0,
            "target_fuel_3h",
        ] = np.nan

    # ==================================================
    # REMOVE INTERNAL COLUMN
    # ==================================================

    df = df.drop(
        columns=["_row_id"]
    )

    # ==================================================
    # FEATURE SUMMARY
    # ==================================================

    original_columns = [
        "timestamp",
        "generator_id",
        "fuel_level_l",
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
        "target_fuel_3h",
    ]

    new_features = [
        column
        for column in df.columns
        if column not in original_columns
    ]

    print(
        f"\nOriginal columns: "
        f"{len(original_columns)}"
    )

    print(
        f"New V2 features: "
        f"{len(new_features)}"
    )

    print("\nNew features:")

    for column in new_features:

        print(
            f"  - {column}"
        )

    # ==================================================
    # SAVE
    # ==================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nDataset shape: "
        f"{df.shape}"
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )

    print("\n" + "=" * 70)
    print("FORECASTING FEATURES V2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()