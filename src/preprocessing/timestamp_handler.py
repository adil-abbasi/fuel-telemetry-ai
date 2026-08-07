import pandas as pd
import numpy as np


class TimestampHandler:
    """
    Timestamp preprocessing module.

    Responsibilities
    ----------------
    1. Detect communication gaps.
    2. Classify gaps.
    3. Rebuild complete timeline.
    4. Insert placeholder rows.
    5. Produce preprocessing report.

    This module NEVER estimates telemetry values.
    """

    SHORT_GAP = 2
    MEDIUM_GAP = 10

    def __init__(self, df: pd.DataFrame):

        self.df = df.copy()

        self.gap_events = []

    # --------------------------------------------------
    # Detect gaps
    # --------------------------------------------------

    def detect_gaps(self):

        self.df = self.df.sort_values(
            ["generator_id", "timestamp"]
        ).reset_index(drop=True)

        self.df["time_difference"] = (

            self.df.groupby("generator_id")["timestamp"]

            .diff()

            .dt.total_seconds()

            / 60

        )

    # --------------------------------------------------
    # Gap classification
    # --------------------------------------------------

    def classify_gap(self, minutes):

        if pd.isna(minutes):
            return None

        if minutes <= self.SHORT_GAP:
            return "Short"

        if minutes <= self.MEDIUM_GAP:
            return "Medium"

        return "Long"

    # --------------------------------------------------
    # Build timeline
    # --------------------------------------------------

    def rebuild_timeline(self):

        rebuilt = []

        telemetry_columns = [

            "status",

            "fuel_level_l",

            "current",

            "battery_voltage"

        ]

        for generator, group in self.df.groupby("generator_id"):

            group = group.sort_values("timestamp")

            start = group["timestamp"].min()

            end = group["timestamp"].max()

            timeline = pd.date_range(

                start=start,

                end=end,

                freq="1min"

            )

            timeline_df = pd.DataFrame({

                "timestamp": timeline

            })

            merged = timeline_df.merge(

                group,

                how="left",

                on="timestamp"

            )

            merged["generator_id"] = generator

            merged["site_name"] = (

                group["site_name"]

                .dropna()

                .iloc[0]

            )

            merged["site_quality"] = (

                group["site_quality"]

                .dropna()

                .iloc[0]

            )

            merged["is_inserted_timestamp"] = (

                merged[telemetry_columns]

                .isna()

                .all(axis=1)

            )

            rebuilt.append(merged)

        self.df = pd.concat(

            rebuilt,

            ignore_index=True

        )

    # --------------------------------------------------
    # Detect inserted gap lengths
    # --------------------------------------------------

    def analyze_gap_events(self):

        reports = []

        for generator, group in self.df.groupby("generator_id"):

            inserted = group["is_inserted_timestamp"]

            change = inserted.ne(inserted.shift()).cumsum()

            for _, block in group.groupby(change):

                if not block["is_inserted_timestamp"].iloc[0]:
                    continue

                missing = len(block)

                if missing <= self.SHORT_GAP:

                    category = "Short"

                elif missing <= self.MEDIUM_GAP:

                    category = "Medium"

                else:

                    category = "Long"

                reports.append({

                    "generator_id": generator,

                    "start": block["timestamp"].min(),

                    "end": block["timestamp"].max(),

                    "missing_minutes": missing,

                    "category": category

                })

        self.gap_events = pd.DataFrame(reports)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    def generate_summary(self):

        summary = []

        for generator, group in self.df.groupby("generator_id"):

            inserted = group["is_inserted_timestamp"].sum()

            total = len(group)

            availability = round(

                (total - inserted) / total * 100,

                2

            )

            summary.append({

                "generator_id": generator,

                "total_rows": total,

                "inserted_rows": inserted,

                "availability": availability

            })

        return pd.DataFrame(summary)

    # --------------------------------------------------
    # Run
    # --------------------------------------------------

    def run(self):

        self.detect_gaps()

        self.rebuild_timeline()

        self.analyze_gap_events()

        summary = self.generate_summary()

        report = {

            "summary": summary,

            "gap_events": self.gap_events

        }

        return self.df, report