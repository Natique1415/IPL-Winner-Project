"""Shared chronological validation and reporting utilities."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import TimeSeriesSplit

from shared_features import FEATURE_COLUMNS, build_feature_dataset

ROOT = Path(__file__).resolve().parent
COMPARISON_DIR = ROOT / "Step-7" / "Model-Comparison"
METRIC_KEYS = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1_score",
    "balanced_accuracy",
]


def load_data_and_split():
    data = build_feature_dataset()
    unique_dates = pd.Index(data["date"].drop_duplicates().sort_values())
    if len(unique_dates) < 10:
        raise ValueError(
            "Not enough distinct match dates for chronological validation."
        )

    test_date_position = int(len(unique_dates) * 0.8)
    test_start_date = unique_dates[test_date_position]
    train_val = data[data["date"] < test_start_date].copy()
    test = data[data["date"] >= test_start_date].copy()
    if train_val.empty or test.empty:
        raise ValueError("Could not create a chronological 80/20 train/test split.")
    return data, train_val, test


def chronological_cv_splits(frame: pd.DataFrame, n_splits: int = 3):
    """Return expanding-window folds, with all matches on a date kept together."""
    dates = pd.Index(frame["date"].drop_duplicates().sort_values())
    if len(dates) <= n_splits + 1:
        raise ValueError(
            "Not enough distinct dates for chronological cross-validation."
        )
    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for train_date_idx, val_date_idx in splitter.split(dates):
        train_dates = dates.take(train_date_idx)
        val_dates = dates.take(val_date_idx)
        train_idx = frame.index[frame["date"].isin(train_dates)].tolist()
        val_idx = frame.index[frame["date"].isin(val_dates)].tolist()
        # Model scripts use iloc positions, so convert labels to positional indices.
        label_to_position = {label: pos for pos, label in enumerate(frame.index)}
        train_idx = [label_to_position[i] for i in train_idx]
        val_idx = [label_to_position[i] for i in val_idx]
        if train_idx and val_idx:
            folds.append((train_idx, val_idx))
    return folds


def score_predictions(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def save_evaluation(
    model_name: str,
    y_true,
    y_pred,
    train_y,
    best_params,
    cv_score,
    data,
    train_val,
    test,
):
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    metrics = score_predictions(y_true, y_pred)
    slug = {
        "Decision Tree": "decision_tree",
        "Naive Bayes": "naive_bayes",
        "KNN": "knn",
    }[model_name]
    metrics_path = COMPARISON_DIR / f"{slug}_metrics.json"
    payload = dict(metrics)
    payload["best_params"] = best_params
    payload["best_cv_balanced_accuracy"] = float(cv_score)
    payload["train_matches"] = int(len(train_val))
    payload["test_matches"] = int(len(test))
    payload["test_start_date"] = str(test["date"].min().date())
    payload["test_end_date"] = str(test["date"].max().date())
    metrics_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1]),
        display_labels=["Team 2 wins", "Team 1 wins"],
    ).plot(ax=axes[0], colorbar=False)
    axes[0].set_title(f"{model_name} : Test Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "Balanced Acc.",
    ]
    values = [metrics[k] for k in METRIC_KEYS]
    bars = axes[1].bar(metric_names, values)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"{model_name} : Test Metrics")
    axes[1].tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    plot_path = COMPARISON_DIR / f"{slug}_metrics.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    baseline_class = pd.Series(train_y).mode().iloc[0]
    baseline_acc = accuracy_score(y_true, [baseline_class] * len(y_true))
    print(f"\n{model_name}")
    print(f"Usable matches: {len(data)}")
    print(f"Training/validation pool: {len(train_val)}")
    print(f"Final test matches: {len(test)}")
    print(
        f"Test date range: {test['date'].min().date()} to {test['date'].max().date()}"
    )
    print(f"Best settings: {best_params}")
    print(f"Mean CV balanced accuracy: {cv_score:.3f}")
    print(f"Majority-class baseline accuracy: {baseline_acc:.3f}")
    print("Test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.3f}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved plot: {plot_path}")
    return payload
