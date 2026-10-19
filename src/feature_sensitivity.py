
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/processed/splits"
REPORT_PATH = ROOT / "reports"

X_train = pd.read_csv(DATA_PATH / "X_train.csv")
X_test = pd.read_csv(DATA_PATH / "X_test.csv")
y_train = pd.read_csv(DATA_PATH / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_PATH / "y_test.csv").squeeze()

# Remove port and TCP-window features
remove_features = [
    column for column in X_train.columns
    if "port" in column.lower()
    or "win_bytes" in column.lower()
]

X_train_reduced = X_train.drop(columns=remove_features)
X_test_reduced = X_test.drop(columns=remove_features)

print("Removed features:", remove_features)
print("Remaining features:", len(X_train_reduced.columns))

models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=15, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():
    print("\nModel:", name)

    model.fit(X_train, y_train)
    original_predictions = model.predict(X_test)

    original_accuracy = accuracy_score(
        y_test, original_predictions
    )
    original_f1 = f1_score(
        y_test, original_predictions, average="macro"
    )

    model.fit(X_train_reduced, y_train)
    reduced_predictions = model.predict(X_test_reduced)

    reduced_accuracy = accuracy_score(
        y_test, reduced_predictions
    )
    reduced_f1 = f1_score(
        y_test, reduced_predictions, average="macro"
    )

    results.append({
        "Model": name,
        "Original Accuracy": round(original_accuracy, 4),
        "Reduced Accuracy": round(reduced_accuracy, 4),
        "Original Macro F1": round(original_f1, 4),
        "Reduced Macro F1": round(reduced_f1, 4)
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_PATH / "feature_sensitivity.csv",
    index=False
)

print("\nFEATURE SENSITIVITY RESULTS")
print(results_df.to_string(index=False))
print("\nResults saved to reports/feature_sensitivity.csv")
