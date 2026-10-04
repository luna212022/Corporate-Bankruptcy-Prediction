"""
Evaluation and visualization module for Corporate Bankruptcy Prediction.
Computes evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC),
generates side-by-side confusion matrices, multi-model ROC curves, and bar plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import get_models, train_model


def evaluate_model(model, X_test, y_test):
    """Evaluates a single model on test data and returns metrics and predictions."""
    y_pred = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "Accuracy": round(acc, 4),
        "Precision (Bankrupt)": round(prec, 4),
        "Recall (Bankrupt)": round(rec, 4),
        "F1-Score (Bankrupt)": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1]
    }
    return metrics, y_pred, y_prob


def evaluate_all(models, X_train, y_train, X_test, y_test):
    """Trains and evaluates all candidate models, returning a summary DataFrame."""
    results = []
    predictions = {}
    prediction_scores = {}

    for name, model in models.items():
        print(f"Training and evaluating: {name}...")
        train_model(model, X_train, y_train)
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
        
        metrics["Model"] = name
        results.append(metrics)
        predictions[name] = y_pred
        prediction_scores[name] = y_prob

    df_results = pd.DataFrame(results)
    # Reorder columns logically
    cols = ["Model", "Accuracy", "Precision (Bankrupt)", "Recall (Bankrupt)", "F1-Score (Bankrupt)", "ROC-AUC", "TN", "FP", "FN", "TP"]
    df_results = df_results[cols]
    return df_results, predictions, prediction_scores


def plot_confusion_matrices(predictions, y_test, output_path: str):
    """Generates and saves a grid of confusion matrices."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    class_labels = ["Solvent", "Bankrupt"]

    for idx, (name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
            xticklabels=class_labels, yticklabels=class_labels, cbar=False
        )
        axes[idx].set_title(f"{name}\nTP: {cm[1, 1]} | FN: {cm[1, 0]}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")

    for unused in range(len(predictions), len(axes)):
        fig.delaxes(axes[unused])

    plt.suptitle("Confusion Matrices Across Models (Test Set: 2,002 Solvent, 99 Bankrupt)", fontsize=15, y=0.98, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices to: {output_path}")


def plot_roc_curves(prediction_scores, y_test, output_path: str):
    """Generates and saves combined ROC curves with AUC annotations."""
    plt.figure(figsize=(10, 7))
    colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#ff7f0e', '#ffbb78', '#9467bd', '#d62728']

    for idx, (name, y_prob) in enumerate(prediction_scores.items()):
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=colors[idx % len(colors)], linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.500)", linewidth=1.5)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves to: {output_path}")


def plot_comparison_bar(comparison_df, output_path: str):
    """Generates and saves a performance comparison bar chart."""
    plot_df = comparison_df[["Model", "Accuracy", "Recall (Bankrupt)", "F1-Score (Bankrupt)", "ROC-AUC"]].set_index("Model")
    ax = plot_df.plot(kind="bar", figsize=(14, 6), width=0.8, colormap="viridis")
    plt.title("Comparative Performance Metrics across Machine Learning Models", fontsize=14, fontweight="bold")
    plt.ylabel("Score", fontsize=12)
    plt.ylim([0, 1.05])
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.legend(loc="lower left", fontsize=11, frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for p in ax.patches:
        h = p.get_height()
        if h > 0.05:
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h / 2),
                        ha="center", va="center", fontsize=8, color="white", fontweight="bold", rotation=90)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved model comparison bar chart to: {output_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_dir = os.path.join(base_dir, "data", "processed")
    res_dir = os.path.join(base_dir, "results")

    print("Loading preprocessed CSV data...")
    X_train = pd.read_csv(os.path.join(proc_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(proc_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(proc_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(proc_dir, "y_test.csv")).values.ravel()

    candidate_models = get_models()
    comp_df, preds, scores = evaluate_all(candidate_models, X_train, y_train, X_test, y_test)

    print("\n=== FINAL MODEL PERFORMANCE COMPARISON ===")
    print(comp_df.to_string(index=False))

    csv_path = os.path.join(res_dir, "model_comparison.csv")
    comp_df.to_csv(csv_path, index=False)
    print(f"Saved comparison CSV to: {csv_path}")

    plot_confusion_matrices(preds, y_test, os.path.join(res_dir, "confusion_matrices.png"))
    plot_roc_curves(scores, y_test, os.path.join(res_dir, "roc_curves.png"))
    plot_comparison_bar(comp_df, os.path.join(res_dir, "model_comparison_bar.png"))
