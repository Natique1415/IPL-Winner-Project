import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Folder containing this script and the JSON metric files.
BASE_DIR = Path(__file__).resolve().parent

model_files = {
    "Decision Tree": "decision_tree_metrics.json",
    "Naive Bayes": "naive_bayes_metrics.json",
    "KNN": "knn_metrics.json",
}

# Keep only numeric evaluation metrics. JSON files may also contain nested
# metadata such as best_params, train_matches, and test date ranges.
metric_keys = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1_score",
    "balanced_accuracy",
]

metric_labels = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1_score": "F1-score",
    "balanced_accuracy": "Balanced accuracy",
}

model_metrics = {}

for model_name, filename in model_files.items():
    file_path = BASE_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing metrics file: {file_path}. Run the classifier scripts first."
        )

    with file_path.open("r", encoding="utf-8") as file:
        saved_data = json.load(file)

    # Support files that store metrics directly or under a "metrics" object.
    if isinstance(saved_data.get("metrics"), dict):
        saved_data = saved_data["metrics"]

    missing = [key for key in metric_keys if key not in saved_data]
    if missing:
        raise ValueError(
            f"{filename} is missing expected metric(s): {', '.join(missing)}"
        )

    model_metrics[model_name] = {
        metric_labels[key]: float(saved_data[key]) for key in metric_keys
    }

# Rows are metrics; columns are models. Ignore non-metric metadata in JSON.
df = pd.DataFrame(model_metrics)

print("\nClassification Model Comparison (%)")
print((df * 100).round(2).to_string())

# Plot grouped bars.
metrics = df.index.tolist()
models = df.columns.tolist()
x = np.arange(len(metrics))
bar_width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

for i, model in enumerate(models):
    values = df[model].to_numpy(dtype=float) * 100
    offset = (i - (len(models) - 1) / 2) * bar_width
    bars = ax.bar(x + offset, values, bar_width, label=model)
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

# Save only the PNG chart; no CSV file is generated.
output_path = BASE_DIR / "classification_comparison.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\nComparison chart saved to: {output_path}")
