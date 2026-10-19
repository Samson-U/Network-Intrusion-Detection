
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent

data_path = ROOT / "data/processed/network_data_cleaned.csv"
output_path = ROOT / "data/processed/splits"

output_path.mkdir(parents=True, exist_ok=True)

# Load processed dataset
df = pd.read_csv(data_path, low_memory=False)

print("PHASE 5: DATA PREPARATION")
print("-" * 35)
print("Original rows:", len(df))

# Remove invalid negative duration values
duration_columns = [
    "Flow Duration",
    "Flow IAT Mean",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Min"
]

for column in duration_columns:
    if column in df.columns:
        df = df[df[column] >= 0]

print("Rows after duration checks:", len(df))

# Separate features and target
X = df.drop(columns=["Label"])
y = df["Label"]

# Group identical feature patterns
groups = pd.util.hash_pandas_object(X, index=False)

# Create grouped train-test split
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]
y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

# Save datasets
X_train.to_csv(output_path / "X_train.csv", index=False)
X_test.to_csv(output_path / "X_test.csv", index=False)
y_train.to_csv(output_path / "y_train.csv", index=False)
y_test.to_csv(output_path / "y_test.csv", index=False)

# Check for overlapping feature patterns
train_hashes = set(
    pd.util.hash_pandas_object(X_train, index=False)
)

test_hashes = set(
    pd.util.hash_pandas_object(X_test, index=False)
)

overlap = train_hashes.intersection(test_hashes)

# Display results
print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

print("\nUnique training patterns:", len(train_hashes))
print("Unique testing patterns:", len(test_hashes))
print("Overlapping patterns:", len(overlap))

if len(overlap) == 0:
    print("\nSplit validation passed.")
else:
    print("\nWarning: Feature-pattern overlap detected.")

print("Split files saved in:", output_path)
