import textwrap
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# Load the original datasets
matches = pd.read_csv("../../matches.csv")

# Remove duplicate records
matches_clean = matches.drop_duplicates().copy()

# Standardize historical franchise names
team_mapping = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

for column in ["team1", "team2", "toss_winner", "winner"]:
    matches_clean[column] = matches_clean[column].replace(team_mapping)

# Standardize venue names so one stadium is not split into several categories.
# The raw data spells the same ground in several ways (with or without the
# city) and a few grounds were renamed, so each variant maps to one name.
venue_mapping = {
    # Kolkata
    "Eden Gardens, Kolkata": "Eden Gardens",
    # Mumbai
    "Wankhede Stadium, Mumbai": "Wankhede Stadium",
    "Brabourne Stadium, Mumbai": "Brabourne Stadium",
    "Dr DY Patil Sports Academy, Mumbai": "Dr DY Patil Sports Academy",
    # Bengaluru
    "M.Chinnaswamy Stadium": "M Chinnaswamy Stadium",
    "M Chinnaswamy Stadium, Bengaluru": "M Chinnaswamy Stadium",
    # Delhi (Feroz Shah Kotla was renamed Arun Jaitley Stadium)
    "Feroz Shah Kotla": "Arun Jaitley Stadium",
    "Arun Jaitley Stadium, Delhi": "Arun Jaitley Stadium",
    # Hyderabad
    "Rajiv Gandhi International Stadium, Uppal": "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Rajiv Gandhi International Stadium",
    # Chennai
    "MA Chidambaram Stadium, Chepauk": "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk, Chennai": "MA Chidambaram Stadium",
    # Jaipur
    "Sawai Mansingh Stadium, Jaipur": "Sawai Mansingh Stadium",
    # Mohali (Punjab Cricket Association Stadium is the IS Bindra Stadium)
    "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh": "Punjab Cricket Association IS Bindra Stadium",
    # Pune (Subrata Roy Sahara Stadium is the Maharashtra Cricket Association Stadium)
    "Maharashtra Cricket Association Stadium, Pune": "Maharashtra Cricket Association Stadium",
    "Subrata Roy Sahara Stadium": "Maharashtra Cricket Association Stadium",
    # Ahmedabad (Sardar Patel Stadium, Motera was renamed Narendra Modi Stadium)
    "Sardar Patel Stadium, Motera": "Narendra Modi Stadium, Ahmedabad",
    # Visakhapatnam
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam": "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
    # Dharamsala
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": "Himachal Pradesh Cricket Association Stadium",
    # Abu Dhabi
    "Zayed Cricket Stadium, Abu Dhabi": "Sheikh Zayed Stadium",
}

venues_before = matches_clean["venue"].nunique()
matches_clean["venue"] = matches_clean["venue"].replace(venue_mapping)
print(
    f"Unique venues: {venues_before} before cleaning, {matches_clean['venue'].nunique()} after"
)

# Keep matches with a known winner among the participating teams
matches_classification = matches_clean.dropna(
    subset=["winner", "date", "team1", "team2", "venue", "toss_winner", "toss_decision"]
).copy()

matches_classification["date"] = pd.to_datetime(
    matches_classification["date"], errors="coerce"
)

matches_classification = matches_classification.dropna(subset=["date"])

matches_classification = matches_classification[
    matches_classification["winner"].isin(
        pd.concat(
            [
                matches_classification["team1"],
                matches_classification["team2"],
            ]
        ).unique()
    )
].copy()

matches_classification = matches_classification.sort_values("date")

# Create the target: 1 if Team 1 wins, otherwise 0
matches_classification["team1_won"] = (
    matches_classification["winner"] == matches_classification["team1"]
).astype(int)

# Store previous results for each team
team_history = defaultdict(list)
prepared_matches = []

# Build features using only matches played before the current date
for date, daily_matches in matches_classification.groupby("date", sort=True):
    for _, match in daily_matches.iterrows():
        team1 = match["team1"]
        team2 = match["team2"]

        history1 = team_history[team1]
        history2 = team_history[team2]

        # Overall and last-5 win rates from earlier matches only (0.5 if none)
        prior_rate1 = sum(history1) / len(history1) if history1 else 0.5
        prior_rate2 = sum(history2) / len(history2) if history2 else 0.5
        recent_rate1 = sum(history1[-5:]) / len(history1[-5:]) if history1 else 0.5
        recent_rate2 = sum(history2[-5:]) / len(history2[-5:]) if history2 else 0.5

        # Who bats first follows from the toss: the toss winner either bats
        # or fields, so the other team gets the opposite.
        toss_decision = str(match["toss_decision"]).strip().lower()
        if toss_decision not in ("bat", "field"):
            raise ValueError(f"Unexpected toss_decision value: {toss_decision!r}")

        team1_won_toss = match["toss_winner"] == team1
        team1_bats_first = int(team1_won_toss == (toss_decision == "bat"))

        prepared_matches.append(
            {
                "date": date,
                "team1": team1,
                "team2": team2,
                "venue": match["venue"],
                "toss_decision": toss_decision,
                "team1_won_toss": int(team1_won_toss),
                "team1_bats_first": team1_bats_first,
                # Differences remove the arbitrary Team 1 / Team 2 slot order:
                # positive means Team 1 is the stronger side
                "prior_win_rate_diff": prior_rate1 - prior_rate2,
                "recent_win_rate_diff": recent_rate1 - recent_rate2,
                "team1_won": int(match["team1_won"]),
            }
        )

    # Update histories only after processing every match on this date
    for _, match in daily_matches.iterrows():
        winner = match["winner"]
        team_history[match["team1"]].append(int(winner == match["team1"]))
        team_history[match["team2"]].append(int(winner == match["team2"]))

data = pd.DataFrame(prepared_matches)

# Team names are not used as features: Team 1 / Team 2 is an arbitrary slot
# order, so the tree would learn noise from it. Team strength comes from the
# win-rate differences instead.
categorical_features = [
    "venue",
    "toss_decision",
]

numeric_features = [
    "team1_won_toss",
    "team1_bats_first",
    "prior_win_rate_diff",
    "recent_win_rate_diff",
]

features = categorical_features + numeric_features

X = data[features]
y = data["team1_won"]

# Train on earlier matches and test on later matches
split_index = int(len(data) * 0.8)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

if X_train.empty or X_test.empty:
    raise ValueError("Not enough matches for a train/test split.")

# Encode categories and preserve numeric performance features
preprocessor = ColumnTransformer(
    [
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numeric", "passthrough", numeric_features),
    ]
)

# Use entropy-based splits and balance both outcome classes
model = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                criterion="entropy",
                max_depth=3,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Train Team 1 win share:", round(y_train.mean(), 3))
print("Train accuracy:", round(accuracy_score(y_train, model.predict(X_train)), 3))
print("Train predicted Team 1 share:", round(model.predict(X_train).mean(), 3))  # type: ignore
print(X_train["prior_win_rate_diff"].describe().round(3))
print(X_test["prior_win_rate_diff"].describe().round(3))

# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

specificity = tn / (tn + fp) if (tn + fp) else 0

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "Specificity": specificity,
    "F1-score": f1,
}

# Compare against always predicting the training majority class
majority_class = y_train.mode()[0]
baseline_predictions = [majority_class] * len(y_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)

print(f"Usable matches: {len(data)}")
print(f"Training matches: {len(X_train)}")
print(f"Testing matches: {len(X_test)}")
print(
    f"Test date range: {data['date'].iloc[split_index].date()} "
    f"to {data['date'].iloc[-1].date()}"
)
print(f"Model accuracy: {accuracy:.3f}")
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

# Plot the confusion matrix and evaluation metrics
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(y_test, y_pred, labels=[0, 1]),
    display_labels=["Team 2 wins", "Team 1 wins"],
).plot(ax=axes[0], cmap="Blues", colorbar=False)

axes[0].set_title("Confusion Matrix")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

axes[1].bar(metrics.keys(), metrics.values())
axes[1].set_ylim(0, 1.12)
axes[1].set_ylabel("Score")
axes[1].set_title("Classification Metrics")
axes[1].tick_params(axis="x", rotation=25)

for index, value in enumerate(metrics.values()):
    axes[1].text(index, value + 0.025, f"{value:.3f}", ha="center")

fig.tight_layout()
fig.savefig("classification_metrics-classification-2.png", dpi=200)
plt.show()
plt.close(fig)


# ---------------------------------------------------------------------------
# Decision tree visualization (matplotlib only, no Graphviz)
# ---------------------------------------------------------------------------

clf = model.named_steps["classifier"]
tree_ = clf.tree_
feature_names = model.named_steps["preprocessor"].get_feature_names_out()
class_names = ["Team 2 wins", "Team 1 wins"]

# Plain-English questions for the yes/no numeric features
FLAG_QUESTIONS = {
    "team1_won_toss": "Did Team 1 win the toss?",
    "team1_bats_first": "Does Team 1 bat first?",
}

# How the win-rate difference features are described
RATE_DIFF_NAMES = {
    "prior_win_rate_diff": "overall win rate",
    "recent_win_rate_diff": "last-5 win rate",
}

# Plain-English questions for the one-hot encoded categorical features
CATEGORY_QUESTIONS = {
    "venue": "Is the venue {value}?",
    "toss_decision": "Did the toss winner choose to {value}?",
}


def numeric_question(name, threshold):
    """Word a numeric split so that "Yes" always means feature > threshold."""
    if name in FLAG_QUESTIONS:
        return FLAG_QUESTIONS[name]

    what = RATE_DIFF_NAMES[name]
    points = threshold * 100  # win rates are fractions, so x100 = percentage points

    if abs(points) < 0.05:
        return f"Is Team 1's {what} higher than Team 2's?"
    if points > 0:
        return (
            f"Is Team 1's {what} more than {points:.1f} "
            f"percentage points above Team 2's?"
        )
    return (
        f"Is Team 1's {what} less than {abs(points):.1f} "
        f"percentage points below Team 2's, or higher?"
    )


def readable_question(feature_name, threshold):
    """Turn an encoded feature and its threshold into a yes/no question.

    Every question is worded so that "Yes" is the sklearn RIGHT branch
    (feature > threshold), which lets us draw Yes on the left every time.
    """
    kind, name = feature_name.split("__", 1)

    if kind == "numeric":
        return numeric_question(name, threshold)

    for column, template in CATEGORY_QUESTIONS.items():
        prefix = f"{column}_"
        if name.startswith(prefix):
            return template.format(value=name[len(prefix) :])

    return name


# Layout constants (1 data unit = 1 inch, so sizes never drift with the tree)
WRAP_WIDTH = 28
BOX_WIDTH = 2.5
LEAF_SPACING = 2.8
LEVEL_GAP = 1.8
TOP_MARGIN = 1.2
BOTTOM_MARGIN = 0.4
FONT_SIZE = 9

# Build the text for every node
labels = {}
for node in range(tree_.node_count):
    if tree_.children_left[node] == tree_.children_right[node]:
        labels[node] = class_names[int(tree_.value[node][0].argmax())]
    else:
        question = readable_question(
            feature_names[tree_.feature[node]], tree_.threshold[node]
        )
        labels[node] = textwrap.fill(question, width=WRAP_WIDTH)

# One box height for every node, based on the tallest label
max_lines = max(text.count("\n") + 1 for text in labels.values())
BOX_HEIGHT = 0.3 + 0.17 * max_lines

# Place nodes: leaves are evenly spaced left to right, parents sit above
# the midpoint of their children. "Yes" (sklearn right child) goes on the left.
positions = {}
next_leaf_x = [LEAF_SPACING / 2]


def place(node, depth=0):
    yes_child = tree_.children_right[node]
    no_child = tree_.children_left[node]

    if yes_child == no_child:
        positions[node] = (next_leaf_x[0], depth)
        next_leaf_x[0] += LEAF_SPACING
        return

    place(yes_child, depth + 1)
    place(no_child, depth + 1)
    positions[node] = (
        (positions[yes_child][0] + positions[no_child][0]) / 2,
        depth,
    )


place(0)

tree_depth = clf.get_depth()
fig_width = max(clf.get_n_leaves() * LEAF_SPACING, 6)
fig_height = TOP_MARGIN + BOX_HEIGHT + tree_depth * LEVEL_GAP + BOTTOM_MARGIN


def to_y(depth):
    """Convert a tree depth to the y coordinate of that row's box centre."""
    return fig_height - TOP_MARGIN - BOX_HEIGHT / 2 - depth * LEVEL_GAP


fig = plt.figure(figsize=(fig_width, fig_height))
ax = fig.add_axes([0, 0, 1, 1])  # type: ignore
ax.set_xlim(0, fig_width)
ax.set_ylim(0, fig_height)
ax.axis("off")

ax.text(
    fig_width / 2,
    fig_height - 0.35,
    "IPL Match Winner - Decision Tree",
    ha="center",
    va="center",
    fontsize=15,
    fontweight="bold",
)
ax.text(
    fig_width / 2,
    fig_height - 0.72,
    "Start at the top. Yes goes left, No goes right.",
    ha="center",
    va="center",
    fontsize=9.5,
    color="#666666",
)

LINE_COLOR = "#7A7A7A"
YES_COLOR = "#2E7D32"
NO_COLOR = "#B3261E"

# Draw connectors first so the boxes sit on top of them
for node in range(tree_.node_count):
    yes_child = tree_.children_right[node]
    no_child = tree_.children_left[node]

    if yes_child == no_child:
        continue

    px, p_depth = positions[node]
    p_bottom = to_y(p_depth) - BOX_HEIGHT / 2
    bar_y = p_bottom - (LEVEL_GAP - BOX_HEIGHT) / 2

    for child, text, color in [
        (yes_child, "Yes", YES_COLOR),
        (no_child, "No", NO_COLOR),
    ]:
        cx, c_depth = positions[child]
        c_top = to_y(c_depth) + BOX_HEIGHT / 2

        # Elbow connector: down from parent, across, then down into the child
        ax.plot(
            [px, px, cx, cx],
            [p_bottom, bar_y, bar_y, c_top],
            color=LINE_COLOR,
            linewidth=1.4,
            solid_joinstyle="miter",
            zorder=1,
        )

        ax.text(
            cx,
            (bar_y + c_top) / 2,
            text,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
            },
            zorder=3,
        )

# Draw the boxes
for node, (x, depth) in positions.items():
    y_pos = to_y(depth)
    is_leaf = tree_.children_left[node] == tree_.children_right[node]

    if is_leaf:
        if labels[node] == "Team 1 wins":
            face, edge = "#CFE3F7", "#2E6DA4"
        else:
            face, edge = "#FBD9BD", "#C0662B"
        weight = "bold"
    else:
        face, edge = "#F4F4F4", "#555555"
        weight = "normal"

    ax.add_patch(
        FancyBboxPatch(
            (x - BOX_WIDTH / 2, y_pos - BOX_HEIGHT / 2),
            BOX_WIDTH,
            BOX_HEIGHT,
            boxstyle="round,pad=0,rounding_size=0.12",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.4,
            zorder=2,
        )
    )
    ax.text(
        x,
        y_pos,
        labels[node],
        ha="center",
        va="center",
        fontsize=FONT_SIZE,
        fontweight=weight,
        linespacing=1.25,
        zorder=4,
    )

fig.savefig("decision_tree-classification-2.png", dpi=200, facecolor="white")
plt.show()
plt.close(fig)

print("\nDecision tree saved to: decision_tree.png")
print("Metrics plot saved to: classification_metrics.png")
