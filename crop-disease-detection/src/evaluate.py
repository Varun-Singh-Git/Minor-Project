"""
evaluate.py
===========
Generates all evaluation outputs for the trained Decision Tree:

  1. Accuracy, Precision, Recall, F1-Score (per class + overall)
  2. Confusion Matrix heatmap → outputs/confusion_matrix.png
  3. Feature Importance bar chart → outputs/feature_importance.png
  4. Correlation heatmap → outputs/correlation_heatmap.png
  5. Decision Tree visualization → outputs/decision_tree_plot.png

Usage:
    python src/evaluate.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving plots
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.tree import plot_tree

# Add src/ to path so imports work
sys.path.insert(0, os.path.dirname(__file__))
from train import load_features, split_data, load_model

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
FEATURES_CSV    = "data/features.csv"
MODEL_PATH      = "models/decision_tree_model.pkl"
OUTPUTS_DIR     = "outputs"

FEATURE_NAMES   = [
    "contrast", "energy", "homogeneity", "correlation",
    "red_mean", "green_mean", "blue_mean",
    "red_std",  "green_std",  "blue_std"
]

os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# 1 — Metrics summary
# ──────────────────────────────────────────────
def print_metrics(y_test, y_pred, class_names):
    """
    Print overall accuracy, precision, recall, F1 and per-class report.
    """
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred, average="weighted")
    f1   = f1_score(y_test, y_pred, average="weighted")

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Accuracy  : {acc  * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec  * 100:.2f}%")
    print(f"  F1-Score  : {f1   * 100:.2f}%")
    print("=" * 55)

    print("\nPer-Class Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))


# ──────────────────────────────────────────────
# 2 — Confusion Matrix
# ──────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, class_names):
    """
    Plot and save a 5×5 confusion matrix heatmap.
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white"
    )
    plt.title("Confusion Matrix — Crop Disease Detection\n(Decision Tree Classifier)", fontsize=13)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Confusion matrix saved → {out_path}")


# ──────────────────────────────────────────────
# 3 — Feature Importance
# ──────────────────────────────────────────────
def plot_feature_importance(clf, feature_names=FEATURE_NAMES):
    """
    Bar chart of feature importances ranked by Gini Gain contribution.
    """
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]    # descending order

    sorted_names = [feature_names[i] for i in indices]
    sorted_vals  = importances[indices]

    colors = ["#2ecc71" if "glcm" not in n else "#3498db"
              for n in sorted_names]
    colors = ["#e74c3c" if i < 2 else "#e67e22" if i < 4 else "#3498db"
              for i in range(len(sorted_names))]

    plt.figure(figsize=(10, 5))
    bars = plt.barh(sorted_names[::-1], sorted_vals[::-1], color=colors[::-1])
    plt.xlabel("Feature Importance (Gini Gain Contribution)", fontsize=11)
    plt.title("Feature Importance — Decision Tree\nCrop Disease Detection System", fontsize=13)
    plt.axvline(x=0.1, color="gray", linestyle="--", alpha=0.5, label="0.1 threshold")

    # Add value labels on bars
    for bar, val in zip(bars, sorted_vals[::-1]):
        plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "feature_importance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Feature importance saved → {out_path}")


# ──────────────────────────────────────────────
# 4 — Correlation Heatmap (EDA)
# ──────────────────────────────────────────────
def plot_correlation_heatmap(csv_path=FEATURES_CSV):
    """
    Pearson correlation heatmap of 10 features including the numeric label.
    """
    df = pd.read_csv(csv_path)

    # Encode label for correlation analysis
    label_map = {cls: i for i, cls in enumerate(df["label"].unique())}
    df["label_encoded"] = df["label"].map(label_map)

    feature_cols = FEATURE_NAMES + ["label_encoded"]
    corr = df[feature_cols].corr()

    plt.figure(figsize=(10, 8))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True   # show lower triangle only

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1,
        center=0,
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Pearson Correlation Heatmap\n10 Features + Class Label", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = os.path.join(OUTPUTS_DIR, "correlation_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Correlation heatmap saved → {out_path}")


# ──────────────────────────────────────────────
# 5 — Decision Tree Visualization
# ──────────────────────────────────────────────
def plot_decision_tree(clf, class_names, feature_names=FEATURE_NAMES, max_depth=4):
    """
    Render the decision tree structure using sklearn's plot_tree.
    Limited to max_depth=4 for readability in the figure.
    """
    fig, ax = plt.subplots(figsize=(20, 10))

    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=max_depth,      # show top 4 levels only
        impurity=True,
        proportion=False,
        ax=ax
    )

    plt.title(
        f"Decision Tree — Top {max_depth} Levels\n"
        "Crop Disease Detection System (Full tree has more levels)",
        fontsize=14
    )
    plt.tight_layout()

    out_path = os.path.join(OUTPUTS_DIR, "decision_tree_plot.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[OK] Decision tree plot saved → {out_path}")


# ──────────────────────────────────────────────
# 6 — Class Distribution Bar Chart
# ──────────────────────────────────────────────
def plot_class_distribution(csv_path=FEATURES_CSV):
    """
    Bar chart showing number of images per disease class.
    """
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts().sort_index()

    colors = ["#27ae60", "#e74c3c", "#f39c12", "#8e44ad", "#2980b9"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 str(val), ha="center", va="bottom", fontweight="bold", fontsize=10)

    plt.title("Class Distribution — DiaMOS Plant Dataset", fontsize=13)
    plt.xlabel("Disease Class", fontsize=11)
    plt.ylabel("Number of Images", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(OUTPUTS_DIR, "class_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Class distribution saved → {out_path}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Crop Disease Detection — Evaluation & Plots")
    print("=" * 55)

    # Load data and model
    X, y, le = load_features(FEATURES_CSV)
    _, X_test, _, y_test = split_data(X, y)
    clf, label_encoder = load_model(MODEL_PATH)

    class_names = list(label_encoder.classes_)
    y_pred = clf.predict(X_test)

    # 1. Metrics
    print_metrics(y_test, y_pred, class_names)

    # 2. Confusion matrix
    plot_confusion_matrix(y_test, y_pred, class_names)

    # 3. Feature importance
    plot_feature_importance(clf)

    # 4. Correlation heatmap
    plot_correlation_heatmap()

    # 5. Decision tree diagram
    plot_decision_tree(clf, class_names)

    # 6. Class distribution
    plot_class_distribution()

    print(f"\n[DONE] All plots saved in '{OUTPUTS_DIR}/' folder.")
