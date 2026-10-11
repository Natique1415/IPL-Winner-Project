from collections import defaultdict
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


CLASS3_REFERENCE_ACCURACY = 0.486

# Load match-level data and remove duplicate records.
matches = pd.read_csv("../../matches.csv")
matches_clean = matches.drop_duplicates().copy()

# Standardize historical franchise names.
team_mapping = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

for column in ["team1", "team2", "toss_winner", "winner"]:
    matches_clean[column] = matches_clean[column].replace(team_mapping)

# Keep matches with valid pre-match fields and a participating winner.
matches_classification = matches_clean.dropna(
    subset=[
        "winner",
        "date",
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
    ]
).copy()

matches_classification["date"] = pd.to_datetime(
    matches_classification["date"], errors="coerce"
)
matches_classification = matches_classification.dropna(subset=["date"])

matches_classification = (
    matches_classification[
        (matches_classification["winner"] == matches_classification["team1"])
        | (matches_classification["winner"] == matches_classification["team2"])
    ]
    .sort_values("date")
    .copy()
)

matches_classification["team1_won"] = (
    matches_classification["winner"] == matches_classification["team1"]
).astype(int)

# Build pre-match features using only earlier matches.
team_history = defaultdict(list)
prepared_matches = []

for date, daily_matches in matches_classification.groupby("date", sort=True):
    for _, match in daily_matches.iterrows():
        team1 = match["team1"]
        team2 = match["team2"]

        history1 = team_history[team1]
        history2 = team_history[team2]

        prior_rate1 = sum(history1) / len(history1) if history1 else 0.5
        prior_rate2 = sum(history2) / len(history2) if history2 else 0.5

        recent1 = history1[-5:]
        recent2 = history2[-5:]
        recent_rate1 = sum(recent1) / len(recent1) if recent1 else 0.5
        recent_rate2 = sum(recent2) / len(recent2) if recent2 else 0.5

        toss_decision = str(match["toss_decision"]).strip().lower()
        if toss_decision not in ("bat", "field"):
            continue

        team1_won_toss = match["toss_winner"] == team1
        team1_bats_first = team1_won_toss == (toss_decision == "bat")

        prepared_matches.append(
            {
                "date": date,
                "team1_won_toss": int(team1_won_toss),
                "team1_bats_first": int(team1_bats_first),
                "prior_win_rate_diff": prior_rate1 - prior_rate2,
                "recent_win_rate_diff": recent_rate1 - recent_rate2,
                "team1_won": int(match["team1_won"]),
            }
        )

    # Update histories after feature creation for every match on this date.
    for _, match in daily_matches.iterrows():
        team_history[match["team1"]].append(int(match["winner"] == match["team1"]))
        team_history[match["team2"]].append(int(match["winner"] == match["team2"]))

data = pd.DataFrame(prepared_matches).sort_values("date").reset_index(drop=True)

features = [
    "team1_won_toss",
    "team1_bats_first",
    "prior_win_rate_diff",
    "recent_win_rate_diff",
]
X = data[features]
y = data["team1_won"]

# Preserve the final 20% as a chronological test set.
test_start = int(len(data) * 0.8)
train_val = data.iloc[:test_start].copy()
test = data.iloc[test_start:].copy()

if train_val.empty or test.empty:
    raise ValueError("Not enough matches for a chronological train/test split.")

# Use the latest 20% of the pre-test period for K selection.
validation_start = int(len(train_val) * 0.8)
train = train_val.iloc[:validation_start].copy()
validation = train_val.iloc[validation_start:].copy()

if train.empty or validation.empty:
    raise ValueError("Not enough matches for a chronological validation split.")

X_train, y_train = train[features], train["team1_won"]
X_val, y_val = validation[features], validation["team1_won"]
X_train_val, y_train_val = train_val[features], train_val["team1_won"]
X_test, y_test = test[features], test["team1_won"]


# Scaling is essential because KNN compares distances between features.
def make_model(k, weights):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=k,
                    weights=weights,
                    metric="minkowski",
                    p=2,
                ),
            ),
        ]
    )


# Choose K and voting strategy on validation data, not the test set.
candidate_ks = [3, 5, 7, 9, 11, 15, 21]
candidate_weights = ["uniform", "distance"]
best_score = -1.0
best_params = None

for k in candidate_ks:
    if k > len(X_train):
        continue

    for weights in candidate_weights:
        candidate = make_model(k, weights)
        candidate.fit(X_train, y_train)
        val_pred = candidate.predict(X_val)
        score = balanced_accuracy_score(y_val, val_pred)

        if score > best_score:
            best_score = score
            best_params = {"n_neighbors": k, "weights": weights}

if best_params is None:
    raise ValueError("No valid KNN configuration could be trained.")

# Refit the selected configuration on all pre-test matches.
model = make_model(
    best_params["n_neighbors"],
    best_params["weights"],
)
model.fit(X_train_val, y_train_val)
y_pred = model.predict(X_test)

# Calculate test metrics.
accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
specificity = tn / (tn + fp) if (tn + fp) else 0.0

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "Specificity": specificity,
    "F1-score": f1,
    "Balanced accuracy": balanced_accuracy,
}

# Save metrics for the Step 7 model comparison.
comparison_metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "specificity": specificity,
    "f1_score": f1,
    "balanced_accuracy": balanced_accuracy,
}

output_path = (
    Path(__file__).resolve().parents[2]
    / "Step-7"
    / "Model-Comparison"
    / "knn_metrics.json"
)
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(comparison_metrics, file, indent=4)

print(f"JSON metrics saved to: {output_path}")

# Compare against a majority-class baseline from the training period.
majority_class = y_train_val.mode()[0]
baseline_predictions = [majority_class] * len(y_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)

print(f"Usable matches: {len(data)}")
print(f"Training matches used for tuning: {len(X_train)}")
print(f"Validation matches: {len(X_val)}")
print(f"Training + validation matches used for final fit: {len(X_train_val)}")
print(f"Testing matches: {len(X_test)}")
print(f"Test date range: {test['date'].min().date()} to {test['date'].max().date()}")
print(f"Selected settings: {best_params}")
print(f"Best validation balanced accuracy: {best_score:.3f}")
print(f"Class-3 reference accuracy: {CLASS3_REFERENCE_ACCURACY:.3f}")
print(f"KNN test accuracy: {accuracy:.3f}")
print(
    "Accuracy change vs class-3: "
    f"{(accuracy - CLASS3_REFERENCE_ACCURACY) * 100:+.2f} percentage points"
)
print(f"Majority-class baseline accuracy: {baseline_accuracy:.3f}")

print("\nActual test outcomes:")
print(y_test.value_counts().rename(index={0: "Team 2 wins", 1: "Team 1 wins"}))

print("\nPredicted test outcomes:")
print(
    pd.Series(y_pred).value_counts().rename(index={0: "Team 2 wins", 1: "Team 1 wins"})
)

print("\nEvaluation metrics:")
for name, value in metrics.items():
    print(f"{name}: {value:.3f}")

# Save the confusion matrix and metrics chart.
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred, labels=[0, 1]),
    display_labels=["Team 2 wins", "Team 1 wins"],
).plot(ax=axes[0], cmap="Blues", colorbar=False)

axes[0].set_title("KNN Confusion Matrix")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

axes[1].bar(metrics.keys(), metrics.values())
axes[1].set_ylim(0, 1.12)
axes[1].set_ylabel("Score")
axes[1].set_title("KNN Classification Metrics")
axes[1].tick_params(axis="x", rotation=25)

for index, value in enumerate(metrics.values()):
    axes[1].text(index, value + 0.025, f"{value:.3f}", ha="center")

fig.tight_layout()
plot_path = "classification_metrics_knn.png"
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"\nMetrics plot saved to: {plot_path}")
