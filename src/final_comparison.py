
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "reports"
FIGURE_PATH = REPORT_PATH / "figures"

FIGURE_PATH.mkdir(parents=True, exist_ok=True)

supervised = pd.read_csv(
    REPORT_PATH / "model_comparison.csv"
)

anomaly = pd.read_csv(
    REPORT_PATH / "anomaly_model_comparison.csv"
)

# Select the supervised model metrics
supervised_results = supervised[
    ["Model", "Accuracy", "Macro F1"]
].copy()

supervised_results.columns = [
    "Model", "Accuracy", "F1-score"
]

# Select the tuned Isolation Forest result
isolation = anomaly[
    anomaly["Method"] == "Isolation Forest"
].iloc[0]

isolation_result = pd.DataFrame([{
    "Model": "Isolation Forest",
    "Accuracy": isolation["Accuracy"],
    "F1-score": isolation["F1-score"]
}])

# Combine results
comparison = pd.concat(
    [supervised_results, isolation_result],
    ignore_index=True
)

comparison.to_csv(
    REPORT_PATH / "final_model_comparison.csv",
    index=False
)

print("FINAL MODEL COMPARISON")
print(comparison.to_string(index=False))

# Plot F1-score comparison
plt.figure(figsize=(10, 6))

plt.bar(
    comparison["Model"],
    comparison["F1-score"]
)

plt.title("Network Intrusion Detection - Model Comparison")
plt.xlabel("Model")
plt.ylabel("F1-score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "final_model_comparison.png",
    dpi=300
)

plt.close()

print("\nSaved:")
print("reports/final_model_comparison.csv")
print("reports/figures/final_model_comparison.png")
