import pandas as pd


def missing_analysis(df):
    """
    Analyze missing values in the dataset.
    """

    print("\n" + "=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)

    missing = df.isnull().sum()

    missing_percent = (
        missing / len(df)
    ) * 100

    result = pd.DataFrame({
        "Missing Values": missing,
        "Percentage (%)": missing_percent.round(2)
    })

    print(result)

    print("=" * 60)

    return result