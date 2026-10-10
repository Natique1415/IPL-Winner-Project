from collections import defaultdict
import json
from math import exp, log
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
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
ALPHA = (
    1  # Laplace smoothing prevents unseen feature values from zeroing a class score.
)

# Load the same match-level source used by the Decision Tree.
matches = pd.read_csv("../../matches.csv")
matches_clean = matches.drop_duplicates().copy()

# Standardize historical franchise names in the working copy.
team_mapping = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

for column in ["team1", "team2", "toss_winner", "winner"]:
    matches_clean[column] = matches_clean[column].replace(team_mapping)

# Keep matches with the fields needed for pre-match prediction.
matches_classification = matches_clean.dropna(
    subset=["winner", "date", "team1", "team2", "venue", "toss_winner", "toss_decision"]
).copy()
matches_classification["date"] = pd.to_datetime(
    matches_classification["date"], errors="coerce"
)
matches_classification = matches_classification.dropna(subset=["date"])

# Exclude ties, abandoned matches, and results outside the two participating teams.
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

# Build each match's features from earlier matches only.
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
            raise ValueError(f"Unexpected toss_decision value: {toss_decision!r}")

        team1_won_toss = match["toss_winner"] == team1
        team1_bats_first = team1_won_toss == (toss_decision == "bat")
        prior_diff = prior_rate1 - prior_rate2
        recent_diff = recent_rate1 - recent_rate2

        # Overall strength uses a predefined +/-5 percentage-point margin.
        if prior_diff > 0.05:
            prior_category = "Team 1 stronger"
        elif prior_diff < -0.05:
            prior_category = "Team 2 stronger"
        else:
            prior_category = "Similar strength"

        # Last-five rates are discrete; compare their direction around zero.
        if recent_diff > 1e-9:
            recent_category = "Team 1 stronger"
        elif recent_diff < -1e-9:
            recent_category = "Team 2 stronger"
        else:
            recent_category = "Similar strength"

        prepared_matches.append(
            {
                "date": date,
                "team1_won_toss": "Yes" if team1_won_toss else "No",
                "team1_bats_first": "Yes" if team1_bats_first else "No",
                "prior_strength": prior_category,
                "recent_strength": recent_category,
                "team1_won": int(match["team1_won"]),
            }
        )

    # Update histories only after features are built for every match on this date.
    for _, match in daily_matches.iterrows():
        winner = match["winner"]
        team_history[match["team1"]].append(int(winner == match["team1"]))
        team_history[match["team2"]].append(int(winner == match["team2"]))

data = pd.DataFrame(prepared_matches).sort_values("date").reset_index(drop=True)

features = [
    "team1_won_toss",
    "team1_bats_first",
    "prior_strength",
    "recent_strength",
]

# Use the same chronological 80/20 split as class-3.py.
split_index = int(len(data) * 0.8)
train = data.iloc[:split_index].copy()
test = data.iloc[split_index:].copy()

if train.empty or test.empty:
    raise ValueError("Not enough matches for a chronological train/test split.")

X_train = train[features]
y_train = train["team1_won"]
X_test = test[features]
y_test = test["team1_won"]

# Define possible values so Laplace smoothing handles categories absent in a class.
feature_values = {
    "team1_won_toss": ["No", "Yes"],
    "team1_bats_first": ["No", "Yes"],
    "prior_strength": ["Team 1 stronger", "Similar strength", "Team 2 stronger"],
    "recent_strength": ["Team 1 stronger", "Similar strength", "Team 2 stronger"],
}
classes = sorted(y_train.unique())
class_counts = y_train.value_counts().to_dict()
class_priors = {cls: class_counts[cls] / len(y_train) for cls in classes}

# Estimate P(feature=value | class) with Laplace smoothing.
conditional_probs = {}
for cls in classes:
    class_rows = X_train[y_train == cls]
    conditional_probs[cls] = {}

    for feature in features:
        counts = class_rows[feature].value_counts().to_dict()
        category_count = len(feature_values[feature])
        denominator = len(class_rows) + ALPHA * category_count

        conditional_probs[cls][feature] = {
            value: (counts.get(value, 0) + ALPHA) / denominator
            for value in feature_values[feature]
        }


def predict_one(row):
    """Choose the class with the largest Naive Bayes posterior score."""
    log_scores = {}

    for cls in classes:
        score = log(class_priors[cls])
        for feature in features:
            probability = conditional_probs[cls][feature][row[feature]]
            score += log(probability)
        log_scores[cls] = score

    predicted_class = max(log_scores, key=log_scores.get)  # pyright: ignore[reportArgumentType, reportCallIssue]

    # Convert log scores to normalized posterior probabilities for reporting.
    max_score = max(log_scores.values())
    exp_scores = {cls: exp(score - max_score) for cls, score in log_scores.items()}
    total = sum(exp_scores.values())
    posterior = {cls: value / total for cls, value in exp_scores.items()}
    return predicted_class, posterior


predictions_and_probs = [predict_one(row) for _, row in X_test.iterrows()]
y_pred = [item[0] for item in predictions_and_probs]
team1_probabilities = [item[1].get(1, 0.0) for item in predictions_and_probs]

# Calculate the same metrics used for the Decision Tree.
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
    / "naive_bayes_metrics.json"
)
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(comparison_metrics, file, indent=4)

print(f"JSON metrics saved to: {output_path}")

# Compare with the training majority-class baseline.
majority_class = y_train.mode()[0]
baseline_predictions = [majority_class] * len(y_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)

print(f"Usable matches: {len(data)}")
print(f"Training matches: {len(train)}")
print(f"Testing matches: {len(test)}")
print(f"Test date range: {test['date'].min().date()} to {test['date'].max().date()}")
print(f"Class-3 reference accuracy: {CLASS3_REFERENCE_ACCURACY:.3f}")
print(f"Naive Bayes test accuracy: {accuracy:.3f}")
print(
    f"Accuracy change vs class-3: {(accuracy - CLASS3_REFERENCE_ACCURACY) * 100:+.2f} percentage points"
)
print(f"Majority-class baseline accuracy: {baseline_accuracy:.3f}")
print(
    f"Mean predicted probability of Team 1 winning: {sum(team1_probabilities) / len(team1_probabilities):.3f}"
)

print("\nClass priors:")
print(f"P(Team 1 wins) = {class_priors.get(1, 0):.3f}")
print(f"P(Team 2 wins) = {class_priors.get(0, 0):.3f}")

print("\nActual test outcomes:")
print(y_test.value_counts().rename(index={0: "Team 2 wins", 1: "Team 1 wins"}))
print("\nPredicted test outcomes:")
print(
    pd.Series(y_pred).value_counts().rename(index={0: "Team 2 wins", 1: "Team 1 wins"})
)
print("\nEvaluation metrics:")
for name, value in metrics.items():
    print(f"{name}: {value:.3f}")

# Save a confusion matrix and metric chart for the report.
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred, labels=[0, 1]),
    display_labels=["Team 2 wins", "Team 1 wins"],
).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Naive Bayes Confusion Matrix")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

axes[1].bar(metrics.keys(), metrics.values())
axes[1].set_ylim(0, 1.12)
axes[1].set_ylabel("Score")
axes[1].set_title("Naive Bayes Classification Metrics")
axes[1].tick_params(axis="x", rotation=25)
for index, value in enumerate(metrics.values()):
    axes[1].text(index, value + 0.025, f"{value:.3f}", ha="center")

fig.tight_layout()
plot_path = "classification_metrics_naive_bayes.png"
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"\nMetrics plot saved to: {plot_path}")
