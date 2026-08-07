"""
==============================================================
TEST SENSOR HEALTH MONITOR
--------------------------------------------------------------
Tests the Sensor Health Monitor module.

Input
-----
data/processed/imputed_telemetry_dataset.csv

Output
------
data/processed/sensor_health_dataset.csv
==============================================================
"""

from pathlib import Path

import pandas as pd

from src.monitoring.sensor_health_monitor import SensorHealthMonitor


# ==========================================================
# PATHS
# ==========================================================

INPUT_FILE = Path(
    "data/processed/imputed_telemetry_dataset.csv"
)

OUTPUT_FILE = Path(
    "data/processed/sensor_health_dataset.csv"
)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("\n" + "=" * 70)
    print("TESTING SENSOR HEALTH MONITOR")
    print("=" * 70)

    # ------------------------------------------------------
    # Load Dataset
    # ------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    print("\nDataset Shape")
    print(df.shape)

    print("\nPreparing timestamps...")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("\nRunning Sensor Health Monitor...")

    monitor = SensorHealthMonitor(df)

    result = monitor.run()
        # ======================================================
    # SENSOR HEALTH REPORT
    # ======================================================

    print("\n" + "=" * 70)
    print("SENSOR HEALTH SUMMARY")
    print("=" * 70)

    print("\nRows Processed:")
    print(len(result))

    print("\nAverage Overall Health:")
    print(
        round(
            result["overall_sensor_health"].mean(),
            2,
        ),
        "%",
    )

    print("\nOverall Health Statistics")
    print(
        result["overall_sensor_health"]
        .describe()
        .round(2)
    )

    print("\nMaintenance Priority Distribution")
    print(
        result["maintenance_priority"]
        .value_counts()
    )

    print("\nFuel Sensor Health")
    print(
        result["fuel_sensor_health"]
        .describe()
        .round(2)
    )

    print("\nCurrent Sensor Health")
    print(
        result["current_sensor_health"]
        .describe()
        .round(2)
    )

    print("\nBattery Sensor Health")
    print(
        result["battery_sensor_health"]
        .describe()
        .round(2)
    )

    print("\nStatus Sensor Health")
    print(
        result["status_sensor_health"]
        .describe()
        .round(2)
    )
        # ======================================================
    # SAMPLE RESULTS
    # ======================================================

    print("\n" + "=" * 70)
    print("SAMPLE RESULTS")
    print("=" * 70)

    columns = [
        "generator_id",
        "status",
        "estimated_status",
        "fuel_sensor_health",
        "current_sensor_health",
        "battery_sensor_health",
        "status_sensor_health",
        "overall_sensor_health",
        "maintenance_priority",
        "recommended_action",
    ]

    existing_columns = [
        col for col in columns
        if col in result.columns
    ]

    print(
        result[existing_columns]
        .head(30)
    )

    # ======================================================
    # SAVE OUTPUT
    # ======================================================

    print("\nSaving output...")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("GENERATED:")
    print(OUTPUT_FILE)
    print("=" * 70)
    # ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()