
import pandas as pd
import numpy as np
import json
from pathlib import Path

# File paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "CICIDS2017-DOS (PB).csv"
REPORT_PATH = ROOT / "reports"

REPORT_PATH.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH, low_memory=False)

print("\nDATASET AUDIT")
print("=" * 40)

# 1. Basic information
print("\n1. DATASET INFORMATION")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Classes:", df["Label"].nunique())
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# 2. Class distribution
print("\n2. CLASS DISTRIBUTION")

class_counts = df["Label"].value_counts()

print(class_counts)
print("\nClass percentages:")
print((df["Label"].value_counts(normalize=True) * 100).round(2))

class_counts.to_csv(REPORT_PATH / "class_distribution.csv")

# 3. Data types and constant columns
print("\n3. COLUMN CHECK")

print("Data types:")
print(df.dtypes.value_counts())

print("\nConstant columns:")
constant_columns = []

for column in df.columns:
    if df[column].nunique(dropna=False) <= 1:
        constant_columns.append(column)
        print(column)

if not constant_columns:
    print("No constant columns found.")

# 4. Numeric quality
print("\n4. NUMERIC QUALITY")

numeric = df.select_dtypes(include=np.number)

infinite_values = int(np.isinf(numeric).sum().sum())
print("Infinite values:", infinite_values)

print("\nColumns with negative values:")
negative_columns = {}

for column in numeric.columns:
    count = int((df[column] < 0).sum())

    if count > 0:
        negative_columns[column] = count
        print(column, ":", count)

print("\nColumns with very large values:")
for column in numeric.columns:
    if df[column].abs().max() > 1e10:
        print(column, ":", df[column].abs().max())

# 5. Timestamp check
print("\n5. TIMESTAMP CHECK")

timestamp_info = {}

if "Timestamp" in df.columns:
    print("Original values:")
    print(df["Timestamp"].head())

    negative_timestamps = int((df["Timestamp"] < 0).sum())

    valid_timestamps = df.loc[
        df["Timestamp"] >= 0, "Timestamp"
    ]

    converted = pd.to_datetime(
        valid_timestamps,
        unit="ms",
        errors="coerce"
    )

    print("Negative timestamps:", negative_timestamps)
    print("Invalid valid-range timestamps:", converted.isna().sum())

    if not converted.dropna().empty:
        print("Earliest:", converted.min())
        print("Latest:", converted.max())

    print("\nNegative timestamps by class:")
    print(df.loc[df["Timestamp"] < 0, "Label"].value_counts())

    timestamp_info = {
        "negative_timestamps": negative_timestamps,
        "invalid_nonnegative_timestamps": int(converted.isna().sum()),
        "earliest_valid_timestamp": str(converted.min()),
        "latest_valid_timestamp": str(converted.max())
    }

# 6. Identifier review
print("\n6. IDENTIFIER COLUMNS")

identifier_columns = [
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Source Port",
    "Destination Port"
]

for column in identifier_columns:
    if column in df.columns:
        print(column, "- unique values:", df[column].nunique())

# 7. Repeated features and conflicting labels
print("\n7. DUPLICATE FEATURE CHECK")

features = df.drop(columns=["Label"])
repeated_features = features.duplicated(keep=False)

print("Rows with repeated features:", int(repeated_features.sum()))

duplicate_data = df.loc[repeated_features]

if len(duplicate_data) > 0:
    label_counts = duplicate_data.groupby(
        list(features.columns), dropna=False
    )["Label"].nunique()

    conflicting_groups = int((label_counts > 1).sum())
else:
    conflicting_groups = 0

print("Conflicting-label feature groups:", conflicting_groups)

# Save duplicate rows for investigation
duplicate_data.to_csv(
    REPORT_PATH / "duplicate_rows.csv",
    index=False
)

# 8. Highly correlated features
print("\n8. CORRELATION CHECK")

excluded_columns = [
    "Label", "Source IP", "Destination IP", "Timestamp"
]

model_features = df.drop(
    columns=excluded_columns,
    errors="ignore"
)

model_numeric = model_features.select_dtypes(include=np.number)

correlation = model_numeric.corr().abs()

high_correlations = []

for i in range(len(correlation.columns)):
    for j in range(i):
        value = correlation.iloc[i, j]

        if value > 0.95:
            high_correlations.append({
                "feature_1": correlation.columns[i],
                "feature_2": correlation.columns[j],
                "correlation": round(float(value), 4)
            })

correlation_df = pd.DataFrame(
    high_correlations,
    columns=["feature_1", "feature_2", "correlation"]
)

correlation_df.to_csv(
    REPORT_PATH / "high_correlations.csv",
    index=False
)

print("Highly correlated pairs:", len(high_correlations))

# 9. Save column information
column_info = pd.DataFrame({
    "column": df.columns,
    "data_type": df.dtypes.astype(str).values,
    "missing_values": df.isnull().sum().values,
    "unique_values": df.nunique(dropna=False).values
})

column_info.to_csv(
    REPORT_PATH / "column_profile.csv",
    index=False
)

# 10. Save audit summary
summary = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "classes": int(df["Label"].nunique()),
    "missing_values": int(df.isnull().sum().sum()),
    "infinite_values": infinite_values,
    "duplicate_rows": int(df.duplicated().sum()),
    "constant_columns": constant_columns,
    "negative_value_columns": negative_columns,
    "timestamp": timestamp_info,
    "repeated_feature_rows": int(repeated_features.sum()),
    "conflicting_label_groups": conflicting_groups,
    "highly_correlated_pairs": len(high_correlations)
}

with open(
    REPORT_PATH / "audit_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

print("\n" + "=" * 40)
print("AUDIT COMPLETED")
print("Reports saved in:", REPORT_PATH)
print("=" * 40)
