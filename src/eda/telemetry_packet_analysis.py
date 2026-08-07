import pandas as pd
from pathlib import Path


class TelemetryPacketAnalyzer:
    """
    Analyze telemetry packet completeness and quality.
    """

    TELEMETRY_COLUMNS = [
        "status",
        "fuel_level_l",
        "current",
        "battery_voltage",
    ]

    def __init__(self, df: pd.DataFrame):
        # Work on a copy so EDA doesn't accidentally modify the original DataFrame
        self.df = df.copy()

    def classify_packets(self):
        """
        Classify each telemetry record as Complete, Partial, or Empty.
        """

        self.df["missing_sensor_count"] = (
            self.df[self.TELEMETRY_COLUMNS]
            .isna()
            .sum(axis=1)
        )

        self.df["packet_type"] = "Complete"

        self.df.loc[
            self.df["missing_sensor_count"] == len(self.TELEMETRY_COLUMNS),
            "packet_type",
        ] = "Empty"

        self.df.loc[
            (self.df["missing_sensor_count"] > 0)
            & (
                self.df["missing_sensor_count"]
                < len(self.TELEMETRY_COLUMNS)
            ),
            "packet_type",
        ] = "Partial"

    def packet_summary(self) -> pd.DataFrame:
        """
        Generate packet summary.
        """

        summary = (
            self.df["packet_type"]
            .value_counts()
            .rename_axis("Packet Type")
            .reset_index(name="Count")
        )

        summary["Percentage (%)"] = (
            summary["Count"] / len(self.df) * 100
        ).round(2)

        return summary

    def partial_packet_analysis(self) -> pd.DataFrame:
        """
        Analyze which sensors are missing in partial packets.
        """

        partial_df = self.df[
            self.df["packet_type"] == "Partial"
        ]

        report = pd.DataFrame({
            "Missing Count": partial_df[
                self.TELEMETRY_COLUMNS
            ].isna().sum(),

            "Percentage (%)": (
                partial_df[
                    self.TELEMETRY_COLUMNS
                ].isna().mean() * 100
            ).round(2)
        })

        return report

    def save_reports(
        self,
        packet_summary: pd.DataFrame,
        partial_report: pd.DataFrame,
    ):

        Path("reports").mkdir(exist_ok=True)

        packet_summary.to_csv(
            "reports/packet_summary.csv",
            index=False,
        )

        partial_report.to_csv(
            "reports/partial_packet_report.csv",
        )

    def run(self):

        print("\n" + "=" * 70)
        print("TELEMETRY PACKET ANALYSIS")
        print("=" * 70)

        self.classify_packets()

        summary = self.packet_summary()

        partial = self.partial_packet_analysis()

        print("\nPacket Summary\n")
        print(summary)

        print("\nPartial Packet Analysis\n")
        print(partial)

        self.save_reports(summary, partial)

        return self.df, summary, partial