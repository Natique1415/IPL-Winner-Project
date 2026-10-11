"""Decision Tree classifier with shared leakage-safe features and chronological tuning."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

from shared_features import FEATURE_COLUMNS
from model_utils import load_data_and_split, chronological_cv_splits, save_evaluation


def main():
    data, train_val, test = load_data_and_split()
    folds = chronological_cv_splits(train_val, n_splits=3)
    grid = [
        {"max_depth": depth, "min_samples_leaf": leaf, "min_samples_split": split,
         "class_weight": weight, "criterion": "entropy", "random_state": 42}
        for depth in [2, 3, 4, 5, 7, None]
        for leaf in [5, 10, 20, 30]
        for split in [2, 10]
        for weight in [None, "balanced"]
    ]

    best_params, best_score = None, -1.0
    for params in grid:
        fold_scores = []
        for train_idx, val_idx in folds:
            train_fold = train_val.iloc[train_idx]
            val_fold = train_val.iloc[val_idx]
            model = DecisionTreeClassifier(**params)
            model.fit(train_fold[FEATURE_COLUMNS], train_fold["team1_won"])
            pred = model.predict(val_fold[FEATURE_COLUMNS])
            fold_scores.append(balanced_accuracy_score(val_fold["team1_won"], pred))
        score = sum(fold_scores) / len(fold_scores)
        if score > best_score:
            best_score, best_params = score, params

    model = DecisionTreeClassifier(**best_params)
    model.fit(train_val[FEATURE_COLUMNS], train_val["team1_won"])
    pred = model.predict(test[FEATURE_COLUMNS])
    save_evaluation(
        "Decision Tree", test["team1_won"], pred, train_val["team1_won"],
        best_params, best_score, data, train_val, test
    )

    # Draw a compact, readable decision tree using human-friendly questions.
    # Display at most three levels, as in the original project visualization.
    from textwrap import fill
    from matplotlib.patches import FancyBboxPatch

    clf = model
    tree = clf.tree_
    class_names = ["Team 2 wins", "Team 1 wins"]
    display_depth = min(3, clf.get_depth())
    feature_labels = {
        "team1_won_toss": "Did Team 1 win the toss?",
        "team1_bats_first": "Does Team 1 bat first?",
        "prior_win_rate_diff": "Team 1 vs Team 2 overall win rate",
        "recent_5_win_rate_diff": "Team 1 vs Team 2 last-5 win rate",
        "recent_10_win_rate_diff": "Team 1 vs Team 2 last-10 win rate",
        "head_to_head_win_rate_diff": "Team 1 head-to-head advantage",
        "venue_win_rate_diff": "Team 1 venue win-rate advantage",
        "avg_runs_scored_diff": "Team 1 average runs-scored advantage",
        "avg_runs_conceded_diff": "Team 1 average runs-conceded advantage",
        "avg_wickets_taken_diff": "Team 1 average wickets-taken advantage",
        "avg_wickets_lost_diff": "Team 1 average wickets-lost advantage",
    }

    def question_for(node):
        feature = FEATURE_COLUMNS[tree.feature[node]]
        threshold = float(tree.threshold[node])
        if feature == "team1_won_toss":
            return "Did Team 1 lose the toss?" if threshold < 0.5 else "Did Team 1 win the toss?"
        if feature == "team1_bats_first":
            return "Does Team 1 bat second?" if threshold < 0.5 else "Does Team 1 bat first?"
        label = feature_labels.get(feature, feature.replace("_", " "))
        # Every numeric split follows sklearn's left-branch condition: value <= threshold.
        return f"Is {label} at or below {threshold:.2f}?"

    positions = {}
    leaf_cursor = [0]
    visible_nodes = []

    def layout(node, depth=0):
        visible_nodes.append((node, depth))
        left, right = tree.children_left[node], tree.children_right[node]
        is_leaf = left == right
        if is_leaf or depth >= display_depth:
            positions[node] = (leaf_cursor[0], depth)
            leaf_cursor[0] += 1
            return
        layout(left, depth + 1)
        layout(right, depth + 1)
        positions[node] = ((positions[left][0] + positions[right][0]) / 2, depth)

    layout(0)
    max_leaves = max(leaf_cursor[0], 1)
    box_width, box_height = 2.7, 0.88
    x_gap, y_gap = 3.1, 1.65
    fig_width = max(10, max_leaves * x_gap + 1.0)
    fig_height = 2.0 + (display_depth + 1) * y_gap
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.8, max_leaves * x_gap - 0.2)
    ax.set_ylim(-0.3, fig_height)
    ax.axis("off")
    ax.text((max_leaves * x_gap + 0.2) / 2, fig_height - 0.32, "IPL Match Winner - Decision Tree",
            ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text((max_leaves * x_gap + 0.2) / 2, fig_height - 0.68,
            "Follow each question from the top; Yes goes left, No goes right.",
            ha="center", va="center", fontsize=9, color="#666666")

    def xy(node):
        x, depth = positions[node]
        return (x * x_gap + 1.0, fig_height - 1.7 - depth * y_gap)

    # Draw connectors behind the nodes. At the display cutoff, show the model's
    # predicted class for the entire hidden subtree.
    for node, depth in visible_nodes:
        left, right = tree.children_left[node], tree.children_right[node]
        if left == right or depth >= display_depth:
            continue
        px, py = xy(node)
        for child in (left, right):
            cx, cy = xy(child)
            ax.plot([px, px, cx], [py - box_height / 2, (py + cy) / 2, (py + cy) / 2],
                    color="#888888", linewidth=1.1, zorder=1)
            ax.plot([cx, cx], [(py + cy) / 2, cy + box_height / 2],
                    color="#888888", linewidth=1.1, zorder=1)
            branch = "Yes" if child == left else "No"
            ax.text(cx, (py + cy) / 2, branch, ha="center", va="center",
                    fontsize=8, color="#2E7D32" if branch == "Yes" else "#B3261E",
                    fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
                    zorder=3)

    for node, depth in visible_nodes:
        x, y = xy(node)
        left, right = tree.children_left[node], tree.children_right[node]
        is_leaf = left == right
        cutoff = depth >= display_depth and not is_leaf
        predicted_class = int(tree.value[node][0].argmax())
        if is_leaf or cutoff:
            label = class_names[predicted_class]
            face, edge, weight = (
                ("#CFE3F7", "#2E6DA4", "bold") if predicted_class == 1
                else ("#FBD9BD", "#C0662B", "bold")
            )
        else:
            label = fill(question_for(node), width=34)
            face, edge, weight = "#F4F4F4", "#555555", "normal"

        # Include a small sample count on the prediction leaves.
        if is_leaf or cutoff:
            samples = int(tree.n_node_samples[node])
            label = f"{label}\n({samples} matches)"
        ax.add_patch(FancyBboxPatch(
            (x - box_width / 2, y - box_height / 2), box_width, box_height,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=face, edgecolor=edge, linewidth=1.2, zorder=2,
        ))
        ax.text(x, y, label, ha="center", va="center", fontsize=8,
                fontweight=weight, wrap=True, zorder=4)

    tree_path = ROOT / "Step-7" / "Model-Comparison" / "decision_tree_structure.png"
    fig.savefig(tree_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)



if __name__ == "__main__":
    main()
