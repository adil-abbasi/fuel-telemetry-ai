import pandas as pd

from config import RAW_DATA
from src.utils.data_loader import load_dataset
from src.preprocessing.generator_state_estimator import (
    GeneratorStateEstimator,
)


def main():

    print("\nLoading dataset...")

    df = load_dataset(RAW_DATA)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    print("Running Generator State Estimator...")

    estimator = GeneratorStateEstimator(df)

    result_df, report = estimator.run()

    print("\n" + "=" * 70)
    print("GENERATOR STATE ESTIMATION")
    print("=" * 70)

    print("\nSummary")
    print(report["summary"].to_string(index=False))

    print("\nEstimated Status Distribution")
    print(
        result_df["estimated_status"]
        .value_counts(dropna=False)
    )

    print("\nReported vs Estimated")
    print(
        pd.crosstab(
            result_df["reported_status"],
            result_df["estimated_status"],
            dropna=False,
        )
    )

    print("\nSample Status Disagreements")

    print(
        report["disagreements"]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()