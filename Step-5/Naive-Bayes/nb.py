"""Categorical Naive Bayes with feature discretisation and chronological tuning."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from shared_features import FEATURE_COLUMNS
from model_utils import load_data_and_split, chronological_cv_splits, save_evaluation

BINARY_FEATURES = ["team1_won_toss", "team1_bats_first"]
CONTINUOUS_FEATURES = [c for c in FEATURE_COLUMNS if c not in BINARY_FEATURES]


def discretize(frame: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Map continuous values into fixed, ordered, non-negative categorical bins."""
    result = frame[FEATURE_COLUMNS].copy()
    if scheme == "coarse":
        common_edges = [-0.5, -0.15, 0.15, 0.5]
        runs_edges = [-2.0, -0.3, 0.3, 2.0]
        wickets_edges = [-1.0, -0.2, 0.2, 1.0]
    else:
        common_edges = [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]
        runs_edges = [-2.0, -0.5, -0.2, 0.2, 0.5, 2.0]
        wickets_edges = [-1.0, -0.3, -0.1, 0.1, 0.3, 1.0]

    for col in CONTINUOUS_FEATURES:
        if "runs" in col:
            edges = runs_edges
        elif "wickets" in col:
            edges = wickets_edges
        else:
            edges = common_edges
        result[col] = np.digitize(
            pd.to_numeric(result[col], errors="coerce").fillna(0).to_numpy(), edges
        ).astype(int)
    for col in BINARY_FEATURES:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).astype(int)
    return result.astype(int)


def main():
    data, train_val, test = load_data_and_split()
    folds = chronological_cv_splits(train_val, n_splits=3)
    best_params, best_score = None, -1.0

    for scheme in ["coarse", "fine"]:
        binned = discretize(data, scheme)
        for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
            fold_scores = []
            for train_idx, val_idx in folds:
                train_fold = train_val.iloc[train_idx]
                val_fold = train_val.iloc[val_idx]
                model = CategoricalNB(alpha=alpha, min_categories=8)
                model.fit(
                    binned.loc[train_fold.index, FEATURE_COLUMNS],
                    train_fold["team1_won"],
                )
                pred = model.predict(binned.loc[val_fold.index, FEATURE_COLUMNS])
                fold_scores.append(balanced_accuracy_score(val_fold["team1_won"], pred))
            score = sum(fold_scores) / len(fold_scores)
            if score > best_score:
                best_score = score
                best_params = {"alpha": alpha, "binning": scheme, "min_categories": 8}

    binned = discretize(data, best_params["binning"])
    model = CategoricalNB(
        alpha=best_params["alpha"], min_categories=best_params["min_categories"]
    )
    model.fit(binned.loc[train_val.index, FEATURE_COLUMNS], train_val["team1_won"])
    pred = model.predict(binned.loc[test.index, FEATURE_COLUMNS])
    save_evaluation(
        "Naive Bayes",
        test["team1_won"],
        pred,
        train_val["team1_won"],
        best_params,
        best_score,
        data,
        train_val,
        test,
    )


if __name__ == "__main__":
    main()
