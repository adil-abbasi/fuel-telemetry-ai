from config import RAW_DATA

import pandas as pd

from src.utils.data_loader import load_dataset
from src.utils.data_validator import validate_columns

from src.eda.dataset_overview import dataset_overview
from src.eda.missing_analysis import missing_analysis
from src.eda.telemetry_packet_analysis import TelemetryPacketAnalyzer
from src.eda.time_series_analysis import TimeSeriesAnalyzer

def main():

    print("\nLoading Dataset...")
    df = load_dataset(RAW_DATA)

    validate_columns(df)

    print("\nConverting Timestamp...")
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    print("\nSorting Dataset...")
    df = (
        df.sort_values(
            ["generator_id", "timestamp"]
        )
        .reset_index(drop=True)
    )

    print("\nDataset Overview...")
    dataset_overview(df)

    print("\nMissing Value Analysis...")
    missing_analysis(df)

    print("\nTelemetry Packet Analysis...")
    packet_analyzer = TelemetryPacketAnalyzer(df)


    df, packet_summary, partial_report = packet_analyzer.run()

    print("\nTimestamp Analysis...")
    time_analyzer = TimeSeriesAnalyzer(df)
    time_analyzer.run()

    gen = df[df["generator_id"] == "Site_17-GEN1"].copy()

    duplicates = gen[
    gen.duplicated(subset=["timestamp"], keep=False)
    ].sort_values("timestamp")

    print(duplicates.head(20))
    print("\nEDA Completed Successfully.")


if __name__ == "__main__":
    main()