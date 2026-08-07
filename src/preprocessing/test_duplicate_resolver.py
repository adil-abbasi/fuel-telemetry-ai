from config import RAW_DATA

from src.utils.data_loader import load_dataset
from src.preprocessing.duplicate_packet_resolver import DuplicatePacketResolver

import pandas as pd


df = load_dataset(RAW_DATA)

df["timestamp"] = pd.to_datetime(df["timestamp"])

resolver = DuplicatePacketResolver(df)

clean_df, summary, conflicts = resolver.run()

print("\nOriginal Rows :", len(df))
print("Resolved Rows :", len(clean_df))

print("\nDuplicate Summary")
print(summary["duplicate_type"].value_counts())

print("\nConflicting Packets")
print(len(conflicts))