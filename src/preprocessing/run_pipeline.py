import pandas as pd
import os
from src.preprocessing.ml_pipeline import TelemetryMLPipeline

def main():
    # 1. Path set karein (Apni actual raw data CSV ka path yahan likhein)
    raw_data_path = "data/raw/telemetry_data.csv" # Is path ko apne hisaab se change karein
    
    print(f"Loading raw data from {raw_data_path}...")
    df_raw = pd.read_csv(raw_data_path)
    
    # 2. Pipeline initialize aur run karein
    pipeline = TelemetryMLPipeline(df_raw)
    train_df, val_df, test_df = pipeline.run_pipeline()
    
    print("\n--- Data Shapes ---")
    print(f"Train Shape: {train_df.shape}")
    print(f"Val Shape:   {val_df.shape}")
    print(f"Test Shape:  {test_df.shape}")
    
    # 3. Processed data ko save karne ke liye folders create karein (agar nahi hain)
    os.makedirs("data/processed", exist_ok=True)
    
    # 4. Data save karein taake ML models direct yahan se read kar sakein
    print("\nSaving processed splits...")
    train_df.to_csv("data/processed/train_data.csv", index=False)
    val_df.to_csv("data/processed/val_data.csv", index=False)
    test_df.to_csv("data/processed/test_data.csv", index=False)
    
    print("Success! Data is processed and saved in 'data/processed/' folder.")

if __name__ == "__main__":
    main()