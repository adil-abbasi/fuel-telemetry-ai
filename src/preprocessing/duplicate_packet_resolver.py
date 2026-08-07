import pandas as pd
from pathlib import Path


class DuplicatePacketResolver:
    """
    Resolve duplicate telemetry packets for the same
    generator at the same timestamp.

    Strategy
    --------
    1. Identify duplicate timestamp groups.
    2. Classify each group.
    3. Merge complementary packets.
    4. Keep one copy of exact duplicates.
    5. Preserve conflicting duplicates for review.
    """

    KEY_COLUMNS = [
        "generator_id",
        "timestamp",
    ]

    TELEMETRY_COLUMNS = [
        "status",
        "fuel_level_l",
        "current",
        "battery_voltage",
    ]

    STATIC_COLUMNS = [
        "site_name",
        "site_quality",
    ]

    def __init__(self, df: pd.DataFrame):

        self.df = df.copy()

        self.df = self.df.sort_values(
            self.KEY_COLUMNS
        ).reset_index(drop=True)

        Path("reports").mkdir(exist_ok=True)

        self.summary = []

        self.conflicting_packets = []

    # ----------------------------------------------------
    # Public
    # ----------------------------------------------------

    def run(self):

        duplicate_mask = self.df.duplicated(
            subset=self.KEY_COLUMNS,
            keep=False
        )

        duplicate_groups = (
            self.df[duplicate_mask]
            .groupby(self.KEY_COLUMNS, sort=False)
        )

        resolved_rows = []

        processed_keys = set()

        for key, group in duplicate_groups:

            processed_keys.add(key)

            packet_type, row = self._resolve_group(group)

            resolved_rows.append(row)

            self.summary.append({

                "generator_id": key[0],

                "timestamp": key[1],

                "duplicate_type": packet_type,

                "records": len(group)

            })

        non_duplicates = self.df[
            ~duplicate_mask
        ]

        resolved_df = pd.concat(

            [

                non_duplicates,

                pd.DataFrame(resolved_rows)

            ],

            ignore_index=True

        )

        resolved_df = resolved_df.sort_values(
            self.KEY_COLUMNS
        ).reset_index(drop=True)

        summary_df = pd.DataFrame(self.summary)

        conflict_df = pd.DataFrame(
            self.conflicting_packets
        )

        summary_df.to_csv(

            "reports/duplicate_summary.csv",

            index=False

        )

        conflict_df.to_csv(

            "reports/conflicting_duplicates.csv",

            index=False

        )

        return resolved_df, summary_df, conflict_df

    # ----------------------------------------------------
    # Resolve One Duplicate Group
    # ----------------------------------------------------

    def _resolve_group(self, group):

        if self._is_exact_duplicate(group):

            merged = group.iloc[0].copy()

            return "Exact", merged

        if self._is_complementary(group):

            merged = self._merge_group(group)

            return "Complementary", merged

        merged = self._merge_group(group)

        self._store_conflict(group)

        return "Conflicting", merged

    # ----------------------------------------------------
    # Exact Duplicate
    # ----------------------------------------------------

    def _is_exact_duplicate(self, group):

        return group.nunique(dropna=False).max() == 1

    # ----------------------------------------------------
    # Complementary Duplicate
    # ----------------------------------------------------

    def _is_complementary(self, group):

        for column in self.TELEMETRY_COLUMNS:

            values = group[column].dropna().unique()

            if len(values) > 1:

                return False

        return True

    # ----------------------------------------------------
    # Merge
    # ----------------------------------------------------

    def _merge_group(self, group):

        merged = group.iloc[0].copy()

        for _, row in group.iloc[1:].iterrows():

            merged = merged.combine_first(row)

        return merged

    # ----------------------------------------------------
    # Save Conflict
    # ----------------------------------------------------

    def _store_conflict(self, group):

        for _, row in group.iterrows():

            self.conflicting_packets.append(

                row.to_dict()

            )