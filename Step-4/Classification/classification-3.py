import textwrap
from collections import defaultdict


import json

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


CLASS2_ACCURACY = (
    0.482  # Accuracy reported by class-2.py; update if its result changes.
)

# Load matches; deliveries.csv is not needed for these match-level features.
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

# Keep rows with the fields needed for pre-match prediction.
matches_classification = matches_clean.dropna(
    subset=["winner", "date", "team1", "team2", "venue", "toss_winner", "toss_decision"]
).copy()
matches_classification["date"] = pd.to_datetime(
    matches_classification["date"], errors="coerce"
)
matches_classification = matches_classification.dropna(subset=["date"])

# Keep only matches whose winner was one of the two participating teams.
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

# Calculate historical team form using only earlier dates.
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
        team1_bats_first = int(team1_won_toss == (toss_decision == "bat"))

        prepared_matches.append(
            {
                "id": match["id"],
                "date": date,
                "team1_won_toss": int(team1_won_toss),
                "team1_bats_first": team1_bats_first,
                "prior_win_rate_diff": prior_rate1 - prior_rate2,
                "recent_win_rate_diff": recent_rate1 - recent_rate2,
                "team1_won": int(match["team1_won"]),
            }
        )

    # Update after every match on this date has had its pre-match features made.
    for _, match in daily_matches.iterrows():
        winner = match["winner"]
        team_history[match["team1"]].append(int(winner == match["team1"]))
        team_history[match["team2"]].append(int(winner == match["team2"]))

data = pd.DataFrame(prepared_matches).sort_values("date").reset_index(drop=True)

# Use the same chronological 80/20 row split as class-2.py for comparison.
features = [
    "team1_won_toss",
    "team1_bats_first",
    "prior_win_rate_diff",
    "recent_win_rate_diff",
]
X = data[features]
y = data["team1_won"]
split_index = int(len(data) * 0.8)
train_val = data.iloc[:split_index].copy()
test = data.iloc[split_index:].copy()

if train_val.empty or test.empty:
    raise ValueError("Not enough matches for a chronological train/test split.")

# Reserve the latest portion of pre-test data for hyperparameter selection.
train_dates = train_val["date"].drop_duplicates().sort_values().reset_index(drop=True)
if len(train_dates) < 3:
    raise ValueError("Not enough distinct dates for chronological validation.")
validation_start = train_dates.iloc[int(len(train_dates) * 0.8)]
inner_train = train_val[train_val["date"] < validation_start]
validation = train_val[train_val["date"] >= validation_start]

if inner_train.empty or validation.empty:
    raise ValueError("Not enough matches for chronological validation.")

# Tune tree size and class weighting using validation data, not the final test set.
parameter_grid = [
    (depth, leaf_size, class_weight)
    for depth in [1, 2, 3, 4, 5]
    for leaf_size in [5, 10, 20]
    for class_weight in [None, "balanced"]
]

best_score = -1.0
best_params = None

for depth, leaf_size, class_weight in parameter_grid:
    candidate = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=depth,
        min_samples_leaf=leaf_size,
        class_weight=class_weight,
        random_state=42,
    )
    candidate.fit(inner_train[features], inner_train["team1_won"])
    validation_predictions = candidate.predict(validation[features])
    score = balanced_accuracy_score(validation["team1_won"], validation_predictions)

    if score > best_score:
        best_score = score
        best_params = {
            "max_depth": depth,
            "min_samples_leaf": leaf_size,
            "class_weight": class_weight,
        }

# Retrain with selected settings on all pre-test matches.
model = Pipeline(
    [
        (
            "classifier",
            DecisionTreeClassifier(
                criterion="entropy",
                random_state=42,
                **best_params,  # type: ignore
            ),
        ),
    ]
)
model.fit(train_val[features], train_val["team1_won"])
y_train = train_val["team1_won"]
y_test = test["team1_won"]
y_pred = model.predict(test[features])

# Compute evaluation metrics.
accuracy = accuracy_score(y_test, y_pred)
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
    "Balanced accuracy": balanced_accuracy_score(y_test, y_pred),
}

# Save metrics for the Step-7 model comparison.
with open("../../Step-7/Model-Comparison/decision_tree_metrics.json", "w") as file:
    json.dump(
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        },
        file,
        indent=4,
    )

majority_class = y_train.mode()[0]
baseline_predictions = [majority_class] * len(y_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)

print(f"Usable matches: {len(data)}")
print(f"Training matches: {len(train_val)}")
print(f"Validation matches: {len(validation)}")
print(f"Testing matches: {len(test)}")
print(f"Test date range: {test['date'].min().date()} to {test['date'].max().date()}")
print(f"Selected settings: {best_params}")
print(f"Best validation balanced accuracy: {best_score:.3f}")
print(f"Classification-2 reference accuracy: {CLASS2_ACCURACY:.3f}")
print(f"Classification-3 test accuracy: {accuracy:.3f}")
print(
    f"Accuracy change vs class-2: {(accuracy - CLASS2_ACCURACY) * 100:+.2f} percentage points"
)
print(f"Majority-class baseline accuracy: {baseline_accuracy:.3f}")
print(f"Balanced test accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
print(f"Training Team 1 win share: {y_train.mean():.3f}")
print(f"Predicted Team 1 share: {y_pred.mean():.3f}")  # type: ignore
print("\nActual test outcomes:")
print(y_test.value_counts().rename(index={0: "Team 2 wins", 1: "Team 1 wins"}))
print("\nPredicted test outcomes:")
print(
    pd.Series(y_pred).value_counts().rename(index={0: "Team 2 wins", 1: "Team 1 wins"})
)
print("\nEvaluation metrics:")
for name, value in metrics.items():
    print(f"{name}: {value:.3f}")

# Save metrics under a separate name so class-2 results remain available.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred, labels=[0, 1]),
    display_labels=["Team 2 wins", "Team 1 wins"],
).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Classification-3 Confusion Matrix")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

axes[1].bar(metrics.keys(), metrics.values())
axes[1].set_ylim(0, 1.12)
axes[1].set_ylabel("Score")
axes[1].set_title("Classification-3 Classification Metrics")
axes[1].tick_params(axis="x", rotation=25)
for index, value in enumerate(metrics.values()):
    axes[1].text(index, value + 0.025, f"{value:.3f}", ha="center")
fig.tight_layout()
metrics_path = "classification_metrics_classification-3.png"
fig.savefig(metrics_path, dpi=200, bbox_inches="tight")
plt.close(fig)

# Draw a compact tree with readable questions; no Graphviz required.
clf = model.named_steps["classifier"]
tree = clf.tree_
class_names = ["Team 2 wins", "Team 1 wins"]


def readable_question(feature, threshold):
    if feature == "team1_won_toss":
        return "Did Team 1 win the toss?"
    if feature == "team1_bats_first":
        return "Does Team 1 bat first?"

    if feature == "prior_win_rate_diff":
        rate_name = "overall win rate"
    else:
        rate_name = "last-5 win rate"

    points = threshold * 100
    if points >= 0.05:
        return f"Is Team 1's {rate_name} more than {points:.1f} percentage points above Team 2's?"
    if points <= -0.05:
        return f"Is Team 1's {rate_name} no more than {abs(points):.1f} percentage points below Team 2's?"
    return f"Is Team 1's {rate_name} higher than Team 2's?"


labels = {}
for node in range(tree.node_count):
    if tree.children_left[node] == tree.children_right[node]:
        labels[node] = class_names[int(tree.value[node][0].argmax())]
    else:
        feature = features[tree.feature[node]]
        labels[node] = textwrap.fill(
            readable_question(feature, tree.threshold[node]), width=29
        )

max_lines = max(label.count("\n") + 1 for label in labels.values())
box_height = 0.35 + 0.18 * max_lines
box_width = 2.8
leaf_spacing = 3.2
level_gap = 1.9
positions = {}
next_leaf_x = [leaf_spacing / 2]


def place(node, depth=0):
    left = tree.children_left[node]
    right = tree.children_right[node]
    if left == right:
        positions[node] = (next_leaf_x[0], depth)
        next_leaf_x[0] += leaf_spacing
        return
    place(right, depth + 1)  # Yes (> threshold) goes left.
    place(left, depth + 1)  # No (<= threshold) goes right.
    positions[node] = ((positions[right][0] + positions[left][0]) / 2, depth)


place(0)
tree_depth = clf.get_depth()
fig_width = max(clf.get_n_leaves() * leaf_spacing, 8)
fig_height = 1.5 + box_height + tree_depth * level_gap + 0.8


def y_coord(depth):
    return fig_height - 1.1 - box_height / 2 - depth * level_gap


fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.add_axes([0, 0, 1, 1])  # type: ignore
ax.set_xlim(0, fig_width)
ax.set_ylim(0, fig_height)
ax.axis("off")
ax.text(
    fig_width / 2,
    fig_height - 0.3,
    "IPL Match Winner - Decision Tree",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
)
ax.text(
    fig_width / 2,
    fig_height - 0.68,
    "Start at the top. Yes goes left, No goes right.",
    ha="center",
    va="center",
    fontsize=9,
    color="#666666",
)

# Draw connecting lines first, then place boxes over them.
for node in range(tree.node_count):
    right = tree.children_right[node]  # Yes branch
    left = tree.children_left[node]  # No branch
    if right == left:
        continue
    px, depth = positions[node]
    parent_bottom = y_coord(depth) - box_height / 2
    child_row_y = y_coord(depth + 1) + box_height / 2
    join_y = (parent_bottom + child_row_y) / 2

    for child, branch_text, branch_color in [
        (right, "Yes", "#2E7D32"),
        (left, "No", "#B3261E"),
    ]:
        cx, _ = positions[child]
        ax.plot(
            [px, px, cx, cx],
            [parent_bottom, join_y, join_y, child_row_y],
            color="#777777",
            linewidth=1.2,
            zorder=1,
        )
        ax.text(
            cx,
            join_y,
            branch_text,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=branch_color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
            zorder=3,
        )

for node, (x, depth) in positions.items():
    y = y_coord(depth)
    is_leaf = tree.children_left[node] == tree.children_right[node]
    if is_leaf and labels[node] == "Team 1 wins":
        face, edge, weight = "#CFE3F7", "#2E6DA4", "bold"
    elif is_leaf:
        face, edge, weight = "#FBD9BD", "#C0662B", "bold"
    else:
        face, edge, weight = "#F4F4F4", "#555555", "normal"

    ax.add_patch(
        FancyBboxPatch(
            (x - box_width / 2, y - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.3,
            zorder=2,
        )
    )
    ax.text(
        x,
        y,
        labels[node],
        ha="center",
        va="center",
        fontsize=8,
        fontweight=weight,
        wrap=True,
        zorder=4,
    )

ax.set_xlim(0, fig_width)
ax.set_ylim(0, fig_height)
tree_path = "decision_tree_classification-3.png"
fig.savefig(tree_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"\nDecision tree saved to: {tree_path}")
print(f"Metrics plot saved to: {metrics_path}")

