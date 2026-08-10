import numpy as np
import pandas as pd

class TelemetryMLPipeline:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _build_regular_timeline(self):
        print("1. Building regular 1-minute timeline...")
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values(['generator_id', 'timestamp'])
        
        processed_dfs = []
        for (site, gen), group in self.df.groupby(['site_name', 'generator_id']):
            group = group.set_index('timestamp')
            
            # Create a continuous 1-minute timeline from start to end
            full_range = pd.date_range(start=group.index.min(), end=group.index.max(), freq='1min')
            resampled = group.reindex(full_range)
            
            # Restore IDs and explicitly flag missing packets
            resampled['site_name'] = site
            resampled['generator_id'] = gen
            resampled['is_missing_packet'] = resampled['fuel_level_l'].isna().astype(int)
            
            resampled = resampled.reset_index().rename(columns={'index': 'timestamp'})
            processed_dfs.append(resampled)
            
        self.df = pd.concat(processed_dfs, ignore_index=True)

    def _create_features(self):
        print("2. Engineering time-aware features...")
        
        # Calculate time gap (in minutes) - though now mostly 1 min due to timeline, 
        # it helps identify consecutive valid readings
        self.df['time_diff_mins'] = self.df.groupby('generator_id')['timestamp'].diff().dt.total_seconds() / 60.0
        
        # Raw fuel difference
        self.df['fuel_diff_raw'] = -self.df.groupby('generator_id')["fuel_level_l"].diff()
        
        # True consumption rate
        self.df["fuel_consumption_rate_per_min"] = np.where(
            (self.df['time_diff_mins'] == 1) & (self.df['is_missing_packet'] == 0),
            self.df['fuel_diff_raw'], 
            np.nan # Keep NaN if gap exists so we don't calculate false spikes
        )
        
        # Refueling flag (e.g., fuel goes UP by more than 5 liters)
        self.df['is_refueling'] = (self.df['fuel_diff_raw'] < -5.0).astype(int)
        
        # Time-based rolling statistics
        temp_df = self.df.set_index('timestamp')
        windows = ['15min', '60min']
        for sensor in ["fuel_level_l", "current", "battery_voltage"]:
            for window in windows:
                rolling = temp_df.groupby('generator_id')[sensor].rolling(window, min_periods=1)
                win_name = window.replace('min', '')
                
                # Assign back to main dataframe
                self.df[f"{sensor}_mean_{win_name}"] = rolling.mean().reset_index(level=[0, 1], drop=True).values
                self.df[f"{sensor}_std_{win_name}"] = rolling.std().reset_index(level=[0, 1], drop=True).values

    def _create_target(self):
        print("3. Generating forecasting target (3-hours ahead)...")
        # Since we enforced a strict 1-minute timeline, 3 hours ahead is EXACTLY 180 rows ahead
        self.df = self.df.sort_values(['generator_id', 'timestamp'])
        
        # target_fuel_3h is what the model will try to predict
        self.df['target_fuel_3h'] = self.df.groupby('generator_id')['fuel_level_l'].shift(-180)

    def _chronological_split(self):
        print("4. Performing 70/15/15 Temporal Split...")
        # Sort entirely by time first
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        total_rows = len(self.df)
        train_end = int(total_rows * 0.70)
        val_end = int(total_rows * 0.85)
        
        train_df = self.df.iloc[:train_end].copy()
        val_df = self.df.iloc[train_end:val_end].copy()
        test_df = self.df.iloc[val_end:].copy()
        
        print(f"  Train: {len(train_df)} rows")
        print(f"  Val:   {len(val_df)} rows")
        print(f"  Test:  {len(test_df)} rows")
        
        return train_df, val_df, test_df

    def run_pipeline(self):
        self._build_regular_timeline()
        self._create_features()
        self._create_target()
        train, val, test = self._chronological_split()
        print("Pipeline execution complete! Data is ready for ML.")
        return train, val, test

# --- Usage Example ---
# df_raw = pd.read_csv("your_telemetry_data.csv")
# pipeline = TelemetryMLPipeline(df_raw)
# train_df, val_df, test_df = pipeline.run_pipeline()