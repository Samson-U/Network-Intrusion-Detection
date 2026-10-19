
import pandas as pd
import time
import json
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/processed/splits"
REPORT_PATH = ROOT / "reports"

X_train = pd.read_csv(DATA_PATH / "X_train.csv")
X_test = pd.read_csv(DATA_PATH / "X_test.csv")
y_train = pd.read_csv(DATA_PATH / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_PATH / "y_test.csv").squeeze()

# Handle invalid numeric values
X_train = X_train.replace([float("inf"), -float("inf")], float("nan"))
X_test = X_test.replace([float("inf"), -float("inf")], float("nan"))

# Fill missing values using training data only
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=15,
        random_state=42
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
    print("\nTraining:", name)

    start = time.time()
    model.fit(X_train, y_train)
    training_time = round(time.time() - start, 2)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "Macro Precision": round(precision, 4),
        "Macro Recall": round(recall, 4),
        "Macro F1": round(f1, 4),
        "Training Time (seconds)": training_time
    })

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(f1, 4))
    print("Training time:", training_time, "seconds")

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(REPORT_PATH / "model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
print("\nResults saved to reports/model_comparison.csv")
