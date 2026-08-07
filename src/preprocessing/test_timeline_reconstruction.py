import pandas as pd

from config import RAW_DATA

from src.utils.data_loader import load_dataset

from src.preprocessing.timeline_reconstruction import TimelineReconstructor


def main():

    print("Loading Dataset...")

    df = load_dataset(RAW_DATA)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print()

    print("Running Timeline Reconstruction...")

    reconstructor = TimelineReconstructor(df)

    reconstructed_df, summary = reconstructor.run()

    print()

    print("=" * 70)

    print("TIMELINE RECONSTRUCTION SUMMARY")

    print("=" * 70)

    print(summary)

    print()

    print("Inserted timestamps:")

    print(
        reconstructed_df[
            reconstructed_df["is_inserted_timestamp"]
        ].head(20)
    )


if __name__ == "__main__":
    main()