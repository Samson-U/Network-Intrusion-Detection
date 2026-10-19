
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/processed/splits"
REPORT_PATH = ROOT / "reports"
FIGURE_PATH = REPORT_PATH / "figures"

FIGURE_PATH.mkdir(parents=True, exist_ok=True)

X_train = pd.read_csv(DATA_PATH / "X_train.csv")
X_test = pd.read_csv(DATA_PATH / "X_test.csv")
y_train = pd.read_csv(DATA_PATH / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_PATH / "y_test.csv").squeeze()

X_train = X_train.replace([float("inf"), -float("inf")], float("nan"))
X_test = X_test.replace([float("inf"), -float("inf")], float("nan"))

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

for name, model in models.items():
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        model
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        xticks_rotation=45,
        cmap="Blues"
    )

    plt.title(name + " Confusion Matrix")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(
        FIGURE_PATH / (filename + "_confusion_matrix.png"),
        dpi=300
    )
    plt.show()
    plt.close()

    # Feature importance for tree-based models
    tree_model = model.named_steps[
        "decisiontreeclassifier"
        if name == "Decision Tree"
        else "randomforestclassifier"
    ]

    importance = pd.Series(
        tree_model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False).head(15)

    importance.to_csv(
        REPORT_PATH / (filename + "_feature_importance.csv")
    )

    importance.sort_values().plot(
        kind="barh", figsize=(9, 6)
    )

    plt.title(name + " - Top 15 Important Features")
    plt.xlabel("Importance")
    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH / (filename + "_feature_importance.png"),
        dpi=300
    )
    plt.show()
    plt.close()

print("Evaluation completed. Reports saved.")
