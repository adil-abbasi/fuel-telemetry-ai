import pandas as pd


class TimelineReconstructor:

    def __init__(self, df: pd.DataFrame):

        self.df = df.copy()

        self.summary = []

    # ---------------------------------------------------------
    # Detect normal reporting interval
    # ---------------------------------------------------------

    def _detect_interval(self, generator_df):

        diffs = (
            generator_df["timestamp"]
            .sort_values()
            .diff()
            .dt.total_seconds()
            .dropna()
        )

        if len(diffs) == 0:
            return 60

        # Ignore very large communication outages
        diffs = diffs[diffs <= 300]

        if len(diffs) == 0:
            return 60

        interval = int(diffs.mode().iloc[0])

        return max(interval, 1)

    # ---------------------------------------------------------
    # Build complete timeline
    # ---------------------------------------------------------

    def _reconstruct_generator(self, generator_df):

        generator = generator_df["generator_id"].iloc[0]

        interval = self._detect_interval(generator_df)

        start = generator_df["timestamp"].min()

        end = generator_df["timestamp"].max()

        expected = pd.date_range(
            start=start,
            end=end,
            freq=f"{interval}s"
        )

        timeline = pd.DataFrame({
            "timestamp": expected
        })

        merged = timeline.merge(
            generator_df,
            on="timestamp",
            how="left"
        )

        merged["generator_id"] = merged["generator_id"].fillna(generator)

        merged["site_name"] = merged["site_name"].ffill().bfill()

        merged["site_quality"] = merged["site_quality"].ffill().bfill()

        merged["is_inserted_timestamp"] = (
            merged["status"].isna()
            &
            merged["fuel_level_l"].isna()
            &
            merged["current"].isna()
            &
            merged["battery_voltage"].isna()
        )

        merged["is_original_record"] = ~merged["is_inserted_timestamp"]

        merged["reporting_interval_sec"] = interval

        self.summary.append({

            "Generator": generator,

            "Original Records": len(generator_df),

            "Expected Records": len(merged),

            "Inserted Records":
                merged["is_inserted_timestamp"].sum(),

            "Reporting Interval (sec)": interval

        })

        return merged

    # ---------------------------------------------------------
    # Run reconstruction
    # ---------------------------------------------------------

    def run(self):

        reconstructed = []

        for _, group in self.df.groupby("generator_id"):

            reconstructed.append(
                self._reconstruct_generator(group)
            )

        final_df = (
            pd.concat(reconstructed)
            .sort_values(["generator_id", "timestamp"])
            .reset_index(drop=True)
        )

        summary = pd.DataFrame(self.summary)

        return final_df, summary
    