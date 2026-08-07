import pandas as pd


def dataset_overview(df):

    print("=" * 60)

    print("DATASET OVERVIEW")

    print("=" * 60)

    print(f"Rows : {len(df):,}")

    print(f"Columns : {len(df.columns)}")

    print(f"Generators : {df.generator_id.nunique()}")

    print(f"Sites : {df.site_name.nunique()}")

    print(
        "Date Range :",
        df.timestamp.min(),
        "to",
        df.timestamp.max(),
    )

    print("=" * 60)

    print()

    print(df.dtypes)