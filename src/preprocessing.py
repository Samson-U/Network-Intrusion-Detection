
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

data_path = ROOT / "data/raw/CICIDS2017-DOS (PB).csv"
output_path = ROOT / "data/processed"

output_path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path, low_memory=False)

print("Original rows:", len(df))

# Clean column names and labels
df.columns = df.columns.str.strip()
df["Label"] = df["Label"].str.strip().str.lower()

# Remove identifier columns from initial model features
df = df.drop(
    columns=["Source IP", "Destination IP", "Timestamp"],
    errors="ignore"
)

# Replace infinite values with missing values
df = df.replace([np.inf, -np.inf], np.nan)

# Keep duplicate records for now
df.to_csv(
    output_path / "network_data_cleaned.csv",
    index=False
)

print("Final rows:", len(df))
print("Final columns:", len(df.columns))
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows retained:", df.duplicated().sum())
print("Processed dataset saved successfully.")
