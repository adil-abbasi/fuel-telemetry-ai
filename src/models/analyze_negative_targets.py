from pathlib import Path
import pandas as pd



DATA_PATH = Path(
    "data/processed/fuel_forecasting_dataset.csv"
)


def main():

    print("=" * 70)
    print("NEGATIVE TARGET INVESTIGATION")
    print("=" * 70)

    print("\nLoading dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    print(f"Total rows: {len(df):,}")

    # --------------------------------------------------
    # Find negative targets
    # --------------------------------------------------

    negative = df[
        df["target_fuel_3h"] < 0
    ].copy()

    print("\n" + "=" * 70)
    print("1. NEGATIVE TARGET SUMMARY")
    print("=" * 70)

    print(
        f"Negative target rows: {len(negative):,}"
    )

    if negative.empty:
        print("\nNo negative targets found.")
        return

    print("\nNegative target values:")
    print(
        negative["target_fuel_3h"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------
    # By generator
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("2. NEGATIVE TARGETS BY GENERATOR")
    print("=" * 70)

    by_generator = (
        negative
        .groupby("generator_id")
        .agg(
            records=("target_fuel_3h", "size"),
            mean_target=("target_fuel_3h", "mean"),
            min_target=("target_fuel_3h", "min"),
            max_target=("target_fuel_3h", "max"),
            mean_fuel=("fuel_level_l", "mean"),
            min_fuel=("fuel_level_l", "min"),
            max_fuel=("fuel_level_l", "max"),
        )
        .sort_values(
            "records",
            ascending=False
        )
    )

    print(
        by_generator.round(3)
    )

    # --------------------------------------------------
    # Sensor behavior
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("3. SENSOR CONDITIONS DURING NEGATIVE TARGETS")
    print("=" * 70)

    sensor_columns = [
        "fuel_level_l",
        "fuel_rate_lph",
        "fuel_delta",
        "current",
        "battery_voltage",
        "running_probability",
        "telemetry_quality_score",
    ]

    available = [
        c for c in sensor_columns
        if c in negative.columns
    ]

    print(
        negative[available]
        .describe()
        .round(3)
        .to_string()
    )

    # --------------------------------------------------
    # Estimated status
    # --------------------------------------------------

    if "estimated_status" in negative.columns:

        print("\n" + "=" * 70)
        print("4. ESTIMATED STATUS")
        print("=" * 70)

        print(
            negative["estimated_status"]
            .value_counts(dropna=False)
        )

    # --------------------------------------------------
    # Reported status
    # --------------------------------------------------

    if "status" in negative.columns:

        print("\n" + "=" * 70)
        print("5. REPORTED STATUS")
        print("=" * 70)

        print(
            negative["status"]
            .value_counts(dropna=False)
        )

    # --------------------------------------------------
    # Negative targets with positive fuel
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("6. NEGATIVE TARGET VS CURRENT FUEL")
    print("=" * 70)

    positive_fuel = negative[
        negative["fuel_level_l"] > 0
    ]

    zero_or_negative_fuel = negative[
        negative["fuel_level_l"] <= 0
    ]

    print(
        f"Negative targets with positive current fuel: "
        f"{len(positive_fuel):,}"
    )

    print(
        f"Negative targets with zero/negative current fuel: "
        f"{len(zero_or_negative_fuel):,}"
    )

    # --------------------------------------------------
    # Generator-specific sensor behavior
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("7. SENSOR CONDITIONS BY GENERATOR")
    print("=" * 70)

    generator_sensor = (
        negative
        .groupby("generator_id")
        .agg(
            records=("target_fuel_3h", "size"),
            avg_fuel=("fuel_level_l", "mean"),
            avg_current=("current", "mean"),
            avg_voltage=("battery_voltage", "mean"),
            avg_running_probability=(
                "running_probability",
                "mean"
            ),
        )
        .sort_values(
            "records",
            ascending=False
        )
    )

    print(
        generator_sensor.round(3)
    )

    # --------------------------------------------------
    # Sample records
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("8. SAMPLE NEGATIVE TARGET RECORDS")
    print("=" * 70)

    sample_columns = [
        "timestamp",
        "generator_id",
        "fuel_level_l",
        "target_fuel_3h",
        "fuel_rate_lph",
        "fuel_delta",
        "current",
        "battery_voltage",
        "estimated_status",
        "running_probability",
    ]

    sample_columns = [
        c for c in sample_columns
        if c in negative.columns
    ]

    print(
        negative[
            sample_columns
        ]
        .head(30)
        .to_string(index=False)
    )

    # --------------------------------------------------
    # Site 13 detailed investigation
    # --------------------------------------------------

    if "Site_13-GEN1" in negative["generator_id"].values:

        print("\n" + "=" * 70)
        print("9. SITE_13-GEN1 INVESTIGATION")
        print("=" * 70)

        site13 = negative[
            negative["generator_id"]
            == "Site_13-GEN1"
        ]

        print(
            f"Negative targets: {len(site13):,}"
        )

        print("\nFuel statistics:")
        print(
            site13["fuel_level_l"]
            .describe()
            .round(3)
        )

        if "estimated_status" in site13.columns:

            print("\nEstimated status:")
            print(
                site13["estimated_status"]
                .value_counts(dropna=False)
            )

        if "current" in site13.columns:

            print("\nCurrent statistics:")
            print(
                site13["current"]
                .describe()
                .round(3)
            )

        if "battery_voltage" in site13.columns:

            print("\nBattery voltage statistics:")
            print(
                site13["battery_voltage"]
                .describe()
                .round(3)
            )

        print("\nFirst 50 Site_13 negative targets:")

        print(
            site13[
                sample_columns
            ]
            .head(50)
            .to_string(index=False)
        )

    # --------------------------------------------------
    # Save diagnostic data
    # --------------------------------------------------

    output_path = Path(
        "data/processed/negative_target_analysis.csv"
    )

    negative.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"\nSaved negative target records:"
        f"\n{output_path}"
    )


if __name__ == "__main__":
    main()