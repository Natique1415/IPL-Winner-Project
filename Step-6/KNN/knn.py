"""KNN classifier with scaled shared features and chronological hyperparameter tuning."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared_features import FEATURE_COLUMNS
from model_utils import load_data_and_split, chronological_cv_splits, save_evaluation


def make_model(k, weights, p):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=k, weights=weights, p=p, metric="minkowski"
        )),
    ])


def main():
    data, train_val, test = load_data_and_split()
    folds = chronological_cv_splits(train_val, n_splits=3)
    best_params, best_score = None, -1.0

    for k in [3, 5, 7, 9, 11, 15, 21, 31, 41]:
        for weights in ["uniform", "distance"]:
            for p in [1, 2]:
                fold_scores = []
                valid = True
                for train_idx, val_idx in folds:
                    train_fold = train_val.iloc[train_idx]
                    val_fold = train_val.iloc[val_idx]
                    if k > len(train_fold):
                        valid = False
                        break
                    model = make_model(k, weights, p)
                    model.fit(train_fold[FEATURE_COLUMNS], train_fold["team1_won"])
                    pred = model.predict(val_fold[FEATURE_COLUMNS])
                    fold_scores.append(balanced_accuracy_score(val_fold["team1_won"], pred))
                if not valid or not fold_scores:
                    continue
                score = sum(fold_scores) / len(fold_scores)
                if score > best_score:
                    best_score = score
                    best_params = {"n_neighbors": k, "weights": weights, "p": p}

    model = make_model(
        best_params["n_neighbors"], best_params["weights"], best_params["p"]
    )
    model.fit(train_val[FEATURE_COLUMNS], train_val["team1_won"])
    pred = model.predict(test[FEATURE_COLUMNS])
    save_evaluation(
        "KNN", test["team1_won"], pred, train_val["team1_won"],
        best_params, best_score, data, train_val, test
    )

    # Distances are calculated in the full feature space; this plot projects
    # the selected test match and its neighbours onto two form-related features.
    classifier = model.named_steps["classifier"]
    scaler = model.named_steps["scaler"]
    sample = test.iloc[[0]]
    distances, neighbor_positions = model.named_steps["classifier"].kneighbors(
        scaler.transform(sample[FEATURE_COLUMNS]),
        n_neighbors=best_params["n_neighbors"],
    )
    neighbors = train_val.iloc[neighbor_positions[0]]
    x_feature, y_feature = "recent_5_win_rate_diff", "avg_runs_scored_diff"
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(train_val[x_feature], train_val[y_feature], alpha=0.25, label="Training matches")
    ax.scatter(neighbors[x_feature], neighbors[y_feature], s=80, edgecolor="black",
               label="Nearest neighbours")
    ax.scatter(sample[x_feature], sample[y_feature], marker="*", s=220, edgecolor="black",
               label="Selected test match")
    for number, (_, row) in enumerate(neighbors.iterrows(), start=1):
        ax.annotate(str(number), (row[x_feature], row[y_feature]), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    actual = "Team 1" if int(sample["team1_won"].iloc[0]) == 1 else "Team 2"
    predicted = "Team 1" if int(pred[0]) == 1 else "Team 2"
    ax.set_title(f"KNN neighbours (full {len(FEATURE_COLUMNS)}-feature distance)\n"
                 f"Actual: {actual}; predicted: {predicted}")
    ax.set_xlabel("Recent 5-match win-rate difference")
    ax.set_ylabel("Historical runs-scored difference (scaled)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "Step-7" / "Model-Comparison" / "knn_nearest_neighbors.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
