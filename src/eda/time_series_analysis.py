import pandas as pd
from pathlib import Path


class TimeSeriesAnalyzer:
    """
    Performs time-series quality analysis for telemetry data.
    """

    def __init__(self, df: pd.DataFrame):

        self.df = df.copy()

        self.df = self.df.sort_values(
            ["generator_id", "timestamp"]
        )

        Path("reports").mkdir(exist_ok=True)

    # ----------------------------------------------------------
    # Generator Summary
    # ----------------------------------------------------------

    def generator_summary(self):

        report = []

        for generator in self.df["generator_id"].unique():

            gen_df = self.df[
                self.df["generator_id"] == generator
            ].copy()

            first_time = gen_df["timestamp"].min()

            last_time = gen_df["timestamp"].max()

            expected_records = (
                int(
                    (
                        last_time - first_time
                    ).total_seconds() / 60
                )
                + 1
            )

            actual_records = len(gen_df)

            duplicate_records = gen_df.duplicated(
                subset="timestamp"
            ).sum()

            missing_records = max(
                expected_records - actual_records,
                0,
            )

            availability = round(
                actual_records
                / expected_records
                * 100,
                2,
            )

            report.append(
                {
                    "Generator": generator,
                    "First Timestamp": first_time,
                    "Last Timestamp": last_time,
                    "Expected Records": expected_records,
                    "Actual Records": actual_records,
                    "Missing Records": missing_records,
                    "Duplicate Records": duplicate_records,
                    "Availability (%)": availability,
                }
            )

        report = pd.DataFrame(report)

        report.to_csv(
            "reports/generator_summary.csv",
            index=False,
        )

        print("\nGenerator Summary\n")

        print(report)

        return report

    # ----------------------------------------------------------
    # Gap Distribution
    # ----------------------------------------------------------

    def gap_distribution(self):

        intervals = (
            self.df.groupby("generator_id")["timestamp"]
            .diff()
            .dt.total_seconds()
        )

        intervals = intervals.dropna()

        bins = [
            0,
            30,
            90,
            150,
            300,
            900,
            float("inf"),
        ]

        labels = [
            "<30 sec",
            "30-90 sec (Normal)",
            "90-150 sec",
            "150-300 sec",
            "300-900 sec",
            ">900 sec",
        ]

        distribution = (
            pd.cut(
                intervals,
                bins=bins,
                labels=labels,
            )
            .value_counts()
            .sort_index()
            .reset_index()
        )

        distribution.columns = [
            "Gap Category",
            "Count",
        ]

        distribution.to_csv(
            "reports/gap_distribution.csv",
            index=False,
        )

        print("\nGap Distribution\n")

        print(distribution)

        return distribution

    # ----------------------------------------------------------
    # Missing Timestamp Report
    # ----------------------------------------------------------

    def missing_timestamp_report(self):

        missing_rows = []

        for generator in self.df["generator_id"].unique():

            gen_df = self.df[
                self.df["generator_id"] == generator
            ].copy()

            gen_df["minute"] = (
                gen_df["timestamp"]
                .dt.floor("min")
            )

            expected = pd.date_range(
                start=gen_df["minute"].min(),
                end=gen_df["minute"].max(),
                freq="1min",
            )

            actual = set(gen_df["minute"])

            missing = sorted(
                set(expected) - actual
            )

            for ts in missing:

                missing_rows.append(
                    {
                        "Generator": generator,
                        "Missing Timestamp": ts,
                    }
                )

        report = pd.DataFrame(missing_rows)

        report.to_csv(
            "reports/missing_timestamps.csv",
            index=False,
        )

        print(
            f"\nMissing timestamps found: {len(report):,}"
        )

        return report

    # ----------------------------------------------------------
    # Communication Availability
    # ----------------------------------------------------------

    def communication_availability(
        self,
        generator_summary,
    ):

        report = generator_summary[
            [
                "Generator",
                "Availability (%)",
            ]
        ]

        report.to_csv(
            "reports/communication_availability.csv",
            index=False,
        )

        print("\nCommunication Availability\n")

        print(report)

        return report

    # ----------------------------------------------------------
    # Gap Events
    # ----------------------------------------------------------

    def gap_events(self):

        events = []

        for generator in self.df["generator_id"].unique():

            gen_df = self.df[
                self.df["generator_id"] == generator
            ].copy()

            gen_df["interval"] = (
                gen_df["timestamp"]
                .diff()
                .dt.total_seconds()
                / 60
            )

            gaps = gen_df[
                gen_df["interval"] > 2
            ]

            for _, row in gaps.iterrows():

                events.append(
                    {
                        "Generator": generator,
                        "Gap End": row["timestamp"],
                        "Gap Duration (min)": round(
                            row["interval"],
                            2,
                        ),
                    }
                )

        report = pd.DataFrame(events)

        report.to_csv(
            "reports/gap_events.csv",
            index=False,
        )

        print(
            f"\nGap Events Found: {len(report)}"
        )

        return report

    # ----------------------------------------------------------
    # Run
    # ----------------------------------------------------------

    def run(self):

        print("\n" + "=" * 70)
        print("TIME SERIES ANALYSIS")
        print("=" * 70)

        summary = self.generator_summary()

        gaps = self.gap_distribution()

        missing = self.missing_timestamp_report()

        availability = self.communication_availability(
            summary
        )

        events = self.gap_events()

        return {
            "summary": summary,
            "gaps": gaps,
            "missing": missing,
            "availability": availability,
            "events": events,
        }