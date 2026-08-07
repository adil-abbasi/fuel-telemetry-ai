"""
Test Reporting Interval Detector

Run:

python -m src.preprocessing.test_reporting_interval

"""


from config import RAW_DATA

from src.utils.data_loader import load_dataset

from src.preprocessing.reporting_interval import (
    ReportingIntervalDetector
)


def main():

    print()

    print("Loading dataset...")

    df = load_dataset(
        RAW_DATA
    )


    print()

    print("Preparing timestamps...")


    df["timestamp"] = (

        df["timestamp"]

        .pipe(
            lambda x:
            __import__("pandas")
            .to_datetime(x)
        )

    )


    print()

    print("Running Reporting Interval Detector...")


    detector = ReportingIntervalDetector(
        df
    )


    summary = detector.run()


    detector.print_report()


    print()

    print("=" * 75)

    print("INTERVAL DICTIONARY")

    print("=" * 75)


    print(

        detector.get_interval_dictionary()

    )


    print()

    print("=" * 75)

    print("GENERATOR DIAGNOSTIC EXAMPLE")

    print("=" * 75)


    first_generator = (

        df["generator_id"]

        .iloc[0]

    )


    print(

        detector.generator_diagnostics(

            first_generator

        )

    )


    detector.save_summary(

        "reporting_interval_summary.csv"

    )


    detector.save_interval_dictionary(

        "generator_intervals.csv"

    )


    print()

    print(
        "Files Generated:"
    )

    print(
        "- reporting_interval_summary.csv"
    )

    print(
        "- generator_intervals.csv"
    )



if __name__ == "__main__":

    main()