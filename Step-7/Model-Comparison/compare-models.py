import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Folder containing this script and the JSON files
BASE_DIR = Path(__file__).resolve().parent

# Load metrics saved by each classifier
model_files = {
    "Decision Tree": "decision_tree_metrics.json",
    "Naive Bayes": "naive_bayes_metrics.json",
    "KNN": "knn_metrics.json",
}

model_metrics = {}

for model_name, filename in model_files.items():
    file_path = BASE_DIR / filename

    with open(file_path, "r") as file:
        model_metrics[model_name] = json.load(file)

# Create a table with metrics as rows and models as columns
df = pd.DataFrame(model_metrics)

print("\nClassification Model Comparison")
print((df * 100).round(2).to_string())


# Plot grouped bars
metrics = df.index.tolist()
models = df.columns.tolist()

x = np.arange(len(metrics))
bar_width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

for i, model in enumerate(models):
    values = df[model].values * 100  # type: ignore

    bars = ax.bar(x + (i - 1) * bar_width, values, bar_width, label=model)

    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

ax.set_title("Comparison of IPL Match Classification Models")
ax.set_xlabel("Evaluation Metric")
ax.set_ylabel("Score (%)")
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=15, ha="right")
ax.set_ylim(0, 110)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()

output_path = BASE_DIR / "classification_comparison.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nComparison chart saved to: {output_path}")
