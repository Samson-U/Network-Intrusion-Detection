
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = ROOT / "data/processed/splits"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load the existing train/test split
X_train = pd.read_csv(SPLIT_DIR / "X_train.csv")
X_test = pd.read_csv(SPLIT_DIR / "X_test.csv")
y_train = pd.read_csv(SPLIT_DIR / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(SPLIT_DIR / "y_test.csv").iloc[:, 0]

print("OFFLINE ATTACK SIMULATION")
print("-" * 40)

# Clean invalid numeric values using training medians
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

medians = X_train.median()
X_train = X_train.fillna(medians).fillna(0)
X_test = X_test.fillna(medians).fillna(0)

# Train the existing supervised model
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Prepare the anomaly detector using benign training records
benign_mask = y_train.astype(str).str.lower().eq("benign")
X_benign = X_train.loc[benign_mask]

baseline, validation = train_test_split(
    X_benign,
    test_size=0.20,
    random_state=42
)

# Exclude port and TCP window features, as in our existing detector
anomaly_columns = [
    col for col in X_train.columns
    if "port" not in col.lower()
    and "win_bytes" not in col.lower()
]

def transform_anomaly_data(data):
    data = data[anomaly_columns]
    return np.sign(data) * np.log1p(np.abs(data))

X_baseline = transform_anomaly_data(baseline)
X_validation = transform_anomaly_data(validation)

isolation_forest = IsolationForest(
    n_estimators=100,
    max_samples=1024,
    max_features=1.0,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)
isolation_forest.fit(X_baseline)

# Higher scores indicate more anomalous records
validation_scores = -isolation_forest.decision_function(
    X_validation
)
threshold = np.quantile(validation_scores, 0.90)

print("Models trained.")
print("Anomaly threshold:", round(threshold, 4))

# Build the simulation from actual held-out records
records = []

def add_record(row, scenario, origin, source_label="Unknown"):
    row_df = pd.DataFrame(
        [row],
        columns=X_train.columns
    )
    prediction = rf.predict(row_df)[0]

    anomaly_row = transform_anomaly_data(row_df)
    score = -isolation_forest.decision_function(anomaly_row)[0]

    records.append({
        "scenario": scenario,
        "origin": origin,
        "source_label": source_label,
        "rf_prediction": prediction,
        "rf_correct": (
            prediction == source_label
            if origin == "Dataset replay"
            else None
        ),
        "anomaly_score": score,
        "anomaly_threshold": threshold,
        "anomaly_flag": bool(score > threshold)
    })

# Replay up to five test records from each class
labels = sorted(y_test.astype(str).unique())

for label in labels:
    indices = y_test[y_test.astype(str) == label].index[:5]

    for idx in indices:
        add_record(
            X_test.loc[idx],
            scenario=f"Replay: {label}",
            origin="Dataset replay",
            source_label=label
        )

# Create synthetic scenarios using benign test records
benign_test = X_test[
    y_test.astype(str).str.lower().eq("benign")
]

if benign_test.empty:
    raise ValueError("No benign test records are available.")

# Use training-set quantiles as reference values
q01 = X_train.quantile(0.01)
q99 = X_train.quantile(0.99)

def set_existing_values(row, values):
    for column, value in values.items():
        if column in row.index and pd.notna(value):
            row[column] = max(0, value)
    return row

scenarios = {
    "Synthetic: High-volume pattern": {
        "Total Fwd Packets": q99.get("Total Fwd Packets"),
        "Total Backward Packets": q99.get("Total Backward Packets"),
        "Flow Packets/s": q99.get("Flow Packets/s"),
        "Flow Bytes/s": q99.get("Flow Bytes/s")
    },
    "Synthetic: Slow-connection pattern": {
        "Flow Duration": q99.get("Flow Duration"),
        "Flow IAT Mean": q99.get("Flow IAT Mean"),
        "Flow IAT Max": q99.get("Flow IAT Max"),
        "Flow Packets/s": q01.get("Flow Packets/s")
    },
    "Synthetic: Large-packet pattern": {
        "Packet Length Mean": q99.get("Packet Length Mean"),
        "Packet Length Max": q99.get("Packet Length Max"),
        "Fwd Packet Length Max": q99.get("Fwd Packet Length Max")
    }
}

for scenario, changes in scenarios.items():
    for _, base_row in benign_test.head(5).iterrows():
        synthetic_row = base_row.copy()
        synthetic_row = set_existing_values(
            synthetic_row,
            changes
        )

        add_record(
            synthetic_row,
            scenario=scenario,
            origin="Synthetic feature scenario",
            source_label="Unknown"
        )

# Save detailed results
results = pd.DataFrame(records)
results.to_csv(
    REPORT_DIR / "simulation_results.csv",
    index=False
)

# Summarize results by scenario
summary = results.groupby(
    ["scenario", "origin"],
    dropna=False
).agg(
    records=("rf_prediction", "size"),
    rf_attack_predictions=(
        "rf_prediction",
        lambda s: sum(str(x).lower() != "benign" for x in s)
    ),
    anomaly_flags=("anomaly_flag", "sum"),
    mean_anomaly_score=("anomaly_score", "mean"),
    rf_accuracy=("rf_correct", "mean")
).reset_index()

summary.to_csv(
    REPORT_DIR / "simulation_summary.csv",
    index=False
)

print("\nSIMULATION SUMMARY")
print(summary.to_string(index=False))
print("\nDetailed results saved to reports/simulation_results.csv")
print("Summary saved to reports/simulation_summary.csv")
print("\nOffline simulation completed. No network traffic was sent.")
