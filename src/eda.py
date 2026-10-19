
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    ROOT / "data/processed/network_data_cleaned.csv",
    low_memory=False
)

output = ROOT / "reports/figures"
output.mkdir(parents=True, exist_ok=True)

# 1. Class distribution
counts = df["Label"].value_counts()

print("CLASS DISTRIBUTION")
print(counts)

counts.plot(kind="bar", figsize=(9, 5))

plt.title("Network Traffic Class Distribution")
plt.xlabel("Traffic Class")
plt.ylabel("Number of Records")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(output / "class_distribution.png", dpi=300)
plt.show()
plt.close()

# 2. Compare network features across traffic classes
features = [
    "Flow Duration",
    "Total Fwd Packets",
    "Flow Bytes/s"
]

for feature in features:
    if feature in df.columns:
        # Use positive values for log-scale visualization
        plot_data = df[df[feature] > 0]

        if plot_data.empty:
            print("No positive values for:", feature)
            continue

        plt.figure(figsize=(10, 5))

        plot_data.boxplot(
            column=feature,
            by="Label",
            showfliers=False
        )

        plt.yscale("log")
        plt.title(feature + " by Traffic Class")
        plt.suptitle("")
        plt.xlabel("Traffic Class")
        plt.ylabel(feature + " (log scale)")
        plt.xticks(rotation=30)
        plt.tight_layout()

        filename = feature.lower().replace(" ", "_")

        plt.savefig(
            output / (filename + ".png"),
            dpi=300
        )

        plt.show()  
        plt.close()


# 3. Correlation heatmap

import numpy as np

numeric_df = df.select_dtypes(include=np.number)

# Select the 15 features with the strongest overall correlations
correlation = numeric_df.corr().abs()

top_features = correlation.mean().sort_values(
    ascending=False
).head(15).index

correlation = numeric_df[top_features].corr()

plt.figure(figsize=(12, 9))

plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")

plt.xticks(
    range(len(top_features)),
    top_features,
    rotation=90
)

plt.yticks(
    range(len(top_features)),
    top_features
)

plt.title("Correlation Heatmap of Selected Network Features")
plt.tight_layout()

plt.savefig(
    output / "correlation_heatmap.png",
    dpi=300
)

plt.show()
plt.close()

print("Correlation heatmap saved.")


print("\nAll EDA charts saved in reports/figures.")
