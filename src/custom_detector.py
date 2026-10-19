
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/processed/splits"
REPORT_PATH = ROOT / "reports"

REPORT_PATH.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(DATA_PATH / "X_train.csv")
X_test = pd.read_csv(DATA_PATH / "X_test.csv")
y_train = pd.read_csv(DATA_PATH / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_PATH / "y_test.csv").squeeze()

remove_features = [
    col for col in X_train.columns
    if "port" in col.lower() or "win_bytes" in col.lower()
]

X_train = X_train.drop(columns=remove_features)
X_test = X_test.drop(columns=remove_features)

X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

X_benign = X_train[y_train == "benign"]

X_baseline, X_validation = train_test_split(
    X_benign,
    test_size=0.2,
    random_state=42
)

actual = (y_test != "benign").astype(int)

# Custom maximum-deviation detector
def transform_data(data):
    return np.sign(data) * np.log1p(np.abs(data))

baseline_transformed = transform_data(X_baseline)
validation_transformed = transform_data(X_validation)
test_transformed = transform_data(X_test)

normal_median = baseline_transformed.median()

normal_iqr = (
    baseline_transformed.quantile(0.75)
    - baseline_transformed.quantile(0.25)
)

normal_iqr[normal_iqr == 0] = 1

def anomaly_score(data):
    transformed = transform_data(data)
    deviations = abs(transformed - normal_median) / normal_iqr
    return deviations.max(axis=1)

custom_validation_scores = anomaly_score(X_validation)
custom_test_scores = anomaly_score(X_test)

custom_threshold = custom_validation_scores.quantile(0.90)

custom_predictions = (
    custom_test_scores > custom_threshold
).astype(int)

# Tuned Isolation Forest
model = IsolationForest(
    n_estimators=100,
    max_samples=1024,
    max_features=1.0,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

model.fit(X_baseline)

validation_scores = -model.score_samples(X_validation)
test_scores = -model.score_samples(X_test)

threshold = np.percentile(validation_scores, 90)

isolation_predictions = (
    test_scores > threshold
).astype(int)

# Compare both methods
methods = [
    ("Custom Maximum Deviation", custom_predictions, custom_threshold),
    ("Isolation Forest", isolation_predictions, threshold)
]

results = []

print("ANOMALY DETECTOR COMPARISON")
print("=" * 60)

for name, predictions, threshold_value in methods:
    accuracy = accuracy_score(actual, predictions)
    precision = precision_score(
        actual, predictions, zero_division=0
    )
    recall = recall_score(
        actual, predictions, zero_division=0
    )
    f1 = f1_score(actual, predictions, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        actual, predictions
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0

    results.append({
        "Method": name,
        "Threshold": threshold_value,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "False Positive Rate": fpr
    })

    print(f"\n{name}")
    print(f"Threshold: {threshold_value:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(actual, predictions))

comparison = pd.DataFrame(results)

comparison.to_csv(
    REPORT_PATH / "anomaly_model_comparison.csv",
    index=False
)

# Save tuned Isolation Forest predictions
prediction_results = pd.DataFrame({
    "Actual Label": y_test.reset_index(drop=True),
    "Actual Attack": actual.reset_index(drop=True),
    "Anomaly Score": test_scores,
    "Predicted Attack": isolation_predictions
})

prediction_results.to_csv(
    REPORT_PATH / "anomaly_detection_results.csv",
    index=False
)


# Analyze detection rate for each attack type

prediction_results["Actual Label"] = y_test.reset_index(drop=True).values
prediction_results["Predicted Attack"] = np.asarray(
    isolation_predictions
)

attack_results = []

for label in sorted(prediction_results["Actual Label"].unique()):
    subset = prediction_results[
        prediction_results["Actual Label"] == label
    ]

    total = len(subset)
    detected = int((subset["Predicted Attack"] == 1).sum())

    if label == "benign":
        correct = int((subset["Predicted Attack"] == 0).sum())
        detection_rate = correct / total if total else 0
        missed = total - correct
    else:
        detection_rate = detected / total if total else 0
        missed = total - detected

    attack_results.append({
        "Label": label,
        "Total Records": total,
        "Detected": detected if label != "benign" else total - missed,
        "Missed": missed,
        "Detection Rate": detection_rate
    })

attack_report = pd.DataFrame(attack_results)

print("\nISOLATION FOREST DETECTION BY ATTACK TYPE")
print("-" * 65)
print(attack_report.to_string(
    index=False,
    formatters={"Detection Rate": "{:.2%}".format}
))

attack_report.to_csv(
    REPORT_PATH / "isolation_forest_attack_analysis.csv",
    index=False
)

print("\nSaved: reports/isolation_forest_attack_analysis.csv")


print("\nSaved comparison to reports/anomaly_model_comparison.csv")
print("Saved predictions to reports/anomaly_detection_results.csv")
