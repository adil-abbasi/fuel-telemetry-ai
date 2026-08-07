REQUIRED_COLUMNS = [
    "timestamp",
    "site_name",
    "generator_id",
    "site_quality",
    "status",
    "fuel_level_l",
    "current",
    "battery_voltage",
]


def validate_columns(df):

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing Columns : {missing}"
        )

    print("Dataset validation passed.")