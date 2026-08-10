from pathlib import Path

import pandas as pd


ORIGINAL_PATH = Path(
    "data/processed/fuel_forecasting_dataset.csv"
)

V2_PATH = Path(
    "data/processed/fuel_forecasting_features_v2.csv"
)


def main():

    print("\n" + "=" * 70)
    print("V2 TARGET INTEGRITY CHECK")
    print("=" * 70)

    print("\nLoading datasets...")

    original = pd.read_csv(
        ORIGINAL_PATH,
        low_memory=False,
    )

    v2 = pd.read_csv(
        V2_PATH,
        low_memory=False,
    )

    # --------------------------------------------------
    # Parse timestamps
    # --------------------------------------------------

    original["timestamp"] = pd.to_datetime(
        original["timestamp"],
        errors="coerce",
    )

    v2["timestamp"] = pd.to_datetime(
        v2["timestamp"],
        errors="coerce",
    )

    print(
        f"\nOriginal rows: {len(original):,}"
    )

    print(
        f"V2 rows:       {len(v2):,}"
    )

    # --------------------------------------------------
    # Overall target comparison
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("1. OVERALL TARGET COMPARISON")
    print("=" * 70)

    for name, df in [
        ("Original", original),
        ("V2", v2),
    ]:

        target = pd.to_numeric(
            df["target_fuel_3h"],
            errors="coerce",
        )

        print(
            f"\n{name}"
        )

        print(
            f"Total rows:       {len(df):,}"
        )

        print(
            f"Valid targets:    {target.notna().sum():,}"
        )

        print(
            f"Missing targets:  {target.isna().sum():,}"
        )

        print(
            f"Negative targets: {(target < 0).sum():,}"
        )

    # --------------------------------------------------
    # Generator comparison
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("2. TARGET AVAILABILITY BY GENERATOR")
    print("=" * 70)

    def generator_summary(df):

        result = (
            df.groupby("generator_id")
            .agg(
                records=(
                    "generator_id",
                    "size",
                ),
                valid_targets=(
                    "target_fuel_3h",
                    lambda x:
                    x.notna().sum(),
                ),
            )
        )

        result["availability_pct"] = (
            result["valid_targets"]
            / result["records"]
            * 100
        )

        return result

    original_summary = (
        generator_summary(original)
    )

    v2_summary = (
        generator_summary(v2)
    )

    comparison = (
        original_summary
        .join(
            v2_summary,
            lsuffix="_original",
            rsuffix="_v2",
        )
    )

    comparison[
        "availability_difference"
    ] = (
        comparison[
            "availability_pct_v2"
        ]
        -
        comparison[
            "availability_pct_original"
        ]
    )

    print(
        comparison.round(2).to_string()
    )

    # --------------------------------------------------
    # Find problematic generators
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("3. GENERATORS WITH AVAILABILITY CHANGES")
    print("=" * 70)

    problematic = comparison[
        comparison[
            "availability_difference"
        ].abs()
        > 1
    ]

    if len(problematic):

        print(
            "\nSignificant differences:"
        )

        print(
            problematic.round(2)
            .to_string()
        )

    else:

        print(
            "\nNo significant availability differences."
        )

    # --------------------------------------------------
    # Site 13 detailed check
    # --------------------------------------------------

    generator = "Site_13-GEN1"

    print("\n" + "=" * 70)
    print(
        f"4. DETAILED CHECK: {generator}"
    )
    print("=" * 70)

    original_13 = original[
        original["generator_id"]
        == generator
    ].copy()

    v2_13 = v2[
        v2["generator_id"]
        == generator
    ].copy()

    print(
        f"\nOriginal rows: {len(original_13):,}"
    )

    print(
        f"V2 rows:       {len(v2_13):,}"
    )

    print(
        "\nOriginal target:"
    )

    print(
        original_13[
            "target_fuel_3h"
        ]
        .describe()
        .round(3)
    )

    print(
        "\nV2 target:"
    )

    print(
        v2_13[
            "target_fuel_3h"
        ]
        .describe()
        .round(3)
    )

    # --------------------------------------------------
    # Compare target values directly
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("5. DIRECT TARGET VALUE COMPARISON")
    print("=" * 70)

    original_key = original[
        [
            "generator_id",
            "timestamp",
            "target_fuel_3h",
        ]
    ].copy()

    v2_key = v2[
        [
            "generator_id",
            "timestamp",
            "target_fuel_3h",
        ]
    ].copy()

    merged = original_key.merge(
        v2_key,
        on=[
            "generator_id",
            "timestamp",
        ],
        how="outer",
        suffixes=(
            "_original",
            "_v2",
            ),
        indicator=True,
    )

    print(
        f"\nMerged rows: {len(merged):,}"
    )

    print(
        "\nRows missing from V2:"
    )

    print(
        (
            merged["_merge"]
            == "left_only"
        ).sum()
    )

    print(
        "\nRows only in V2:"
    )

    print(
        (
            merged["_merge"]
            == "right_only"
        ).sum()
    )

    # --------------------------------------------------
    # Target changes
    # --------------------------------------------------

    both = merged[
        merged["_merge"]
        == "both"
    ].copy()

    target_original = pd.to_numeric(
        both[
            "target_fuel_3h_original"
        ],
        errors="coerce",
    )

    target_v2 = pd.to_numeric(
        both[
            "target_fuel_3h_v2"
        ],
        errors="coerce",
    )

    both["target_difference"] = (
        target_v2
        -
        target_original
    )

    changed = both[
        (
            both[
                "target_difference"
            ]
            .abs()
            > 0.000001
        )
    ]

    print(
        f"\nTarget values changed: "
        f"{len(changed):,}"
    )

    if len(changed):

        print(
            "\nLargest target changes:"
        )

        print(
            changed[
                [
                    "generator_id",
                    "timestamp",
                    "target_fuel_3h_original",
                    "target_fuel_3h_v2",
                    "target_difference",
                ]
            ]
            .sort_values(
                "target_difference",
                key=lambda x:
                x.abs(),
                ascending=False,
            )
            .head(20)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------
    # Missing target transitions
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("6. TARGET AVAILABILITY TRANSITIONS")
    print("=" * 70)

    original_valid = (
        both[
            "target_fuel_3h_original"
        ].notna()
    )

    v2_valid = (
        both[
            "target_fuel_3h_v2"
        ].notna()
    )

    original_valid_v2_missing = (
        original_valid
        & ~v2_valid
    )

    original_missing_v2_valid = (
        ~original_valid
        & v2_valid
    )

    print(
        "\nValid in original but missing in V2:"
    )

    print(
        int(
            original_valid_v2_missing.sum()
        )
    )

    print(
        "\nMissing in original but valid in V2:"
    )

    print(
        int(
            original_missing_v2_valid.sum()
        )
    )

    if original_valid_v2_missing.any():

        print(
            "\nExamples:"
        )

        examples = both[
            original_valid_v2_missing
        ]

        print(
            examples[
                [
                    "generator_id",
                    "timestamp",
                    "target_fuel_3h_original",
                    "target_fuel_3h_v2",
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("TARGET INTEGRITY RESULT")
    print("=" * 70)

    if (
        len(changed) == 0
        and
        original_valid_v2_missing.sum() == 0
        and
        original_missing_v2_valid.sum() == 0
    ):

        print(
            "\n[PASS] V2 preserved all target values."
        )

    else:

        print(
            "\n[WARNING] Target integrity differences detected."
        )

        print(
            "Do NOT train V2 until the differences are understood."
        )

    print("\n" + "=" * 70)
    print("TARGET INTEGRITY CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()