"""
Visualization utilities for model evaluation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve


def plot_training_history(
    histories: dict[str, tf.keras.callbacks.History],
    save_path: Path | str | None = None,
):
    """
    Plot training curves for multiple models.
    Args:
        histories: {'model_name': history, ...}
        save_path: Path to save figure (None = show only)
    """
    # Check if any history has AUC metrics
    has_auc = any("auc" in hist.history for hist in histories.values())

    if has_auc:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        axes = [axes]  # Make it iterable

    colors = plt.cm.Set2(range(len(histories)))

    for idx, (name, history) in enumerate(histories.items()):
        color = colors[idx]
        hist = history.history

        # Loss (always present)
        axes[0].plot(hist["loss"], label=f"{name} - Train", color=color, linestyle="-")
        axes[0].plot(
            hist["val_loss"], label=f"{name} - Val", color=color, linestyle="--"
        )

        # AUC (only if present)
        if has_auc and "auc" in hist:
            axes[1].plot(
                hist["auc"], label=f"{name} - Train", color=color, linestyle="-"
            )
            axes[1].plot(
                hist["val_auc"], label=f"{name} - Val", color=color, linestyle="--"
            )

    # Configure Loss plot
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Configure AUC plot (if present)
    if has_auc:
        axes[1].set_title("AUC")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        # Create directory if it doesn't exist
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_comparison_metrics(
    results: dict[str, dict[str, float]], save_path: Path | str | None = None
):
    """
    Plot comparison bar charts and ROC curves.

    Works with both pair-level and voting-level evaluation results.

    Args:
        results: Results from ModelComparator.evaluate_all() or evaluate_all_by_voting()
        save_path: Path to save figure (None = show only)
    """
    model_names = list(results.keys())
    colors = plt.cm.Set2(range(len(model_names)))

    # Check what type of results we have
    has_distances = all("distances" in results[name] for name in model_names)
    has_auc = all("auc" in results[name] for name in model_names)
    has_proba = all("y_pred_proba" in results[name] for name in model_names)

    # Determine layout based on available data
    if has_distances and has_proba:
        # Full layout: metrics, ROC, distances, table
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        layout = "full"
    else:
        # Simple layout: metrics and table only
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes = [[axes[0], None], [None, axes[1]]]  # Fake 2x2 structure
        layout = "simple"

    # 1. Bar chart of metrics (always present)
    if has_auc:
        metrics = ["accuracy", "auc", "precision", "recall", "f1"]
    else:
        metrics = ["accuracy", "precision", "recall", "f1"]

    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    ax = axes[0][0]
    for i, (name, color) in enumerate(zip(model_names, colors)):
        values = [results[name].get(m, 0.0) for m in metrics]
        ax.bar(x + i * width, values, width, label=name, color=color, alpha=0.8)

    ax.set_ylabel("Score")
    ax.set_title("Metrics Comparison")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([m.upper() for m in metrics], rotation=45)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)

    # 2. ROC Curves (only if we have probabilities)
    if layout == "full" and has_proba:
        ax = axes[0][1]
        for name, color in zip(model_names, colors):
            y_true = results[name]["y_true"]
            y_proba = results[name]["y_pred_proba"]

            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = results[name].get("auc", 0.0)

            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves")
        ax.legend()
        ax.grid(alpha=0.3)

    # 3. Distance distributions (only if we have distances)
    if layout == "full" and has_distances:
        ax = axes[1][0]
        for name, color in zip(model_names, colors):
            distances = results[name]["distances"]
            y_true = results[name]["y_true"]

            # Separate by class
            dist_pos = distances[y_true == 1]
            dist_neg = distances[y_true == 0]

            ax.hist(
                dist_pos, bins=50, alpha=0.5, label=f"{name} - Same author", color=color
            )
            ax.hist(
                dist_neg,
                bins=50,
                alpha=0.5,
                label=f"{name} - Different authors",
                color=color,
                linestyle="--",
                edgecolor="black",
            )

        ax.set_xlabel("Euclidean Distance")
        ax.set_ylabel("Frequency")
        ax.set_title("Distance Distributions")
        ax.legend()
        ax.grid(alpha=0.3)

    # 4. Summary table (always present)
    if layout == "full":
        ax = axes[1][1]
    else:
        ax = axes[1]

    ax.axis("off")

    table_data = []
    for name in model_names:
        if has_auc:
            row = [
                name,
                f"{results[name]['accuracy']:.3f}",
                f"{results[name]['auc']:.3f}",
                f"{results[name]['f1']:.3f}",
            ]
        else:
            row = [
                name,
                f"{results[name]['accuracy']:.3f}",
                f"{results[name]['precision']:.3f}",
                f"{results[name]['f1']:.3f}",
            ]
        table_data.append(row)

    if has_auc:
        col_labels = ["Model", "Accuracy", "AUC", "F1"]
    else:
        col_labels = ["Model", "Accuracy", "Precision", "F1"]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Highlight best model
    best_idx = max(enumerate(model_names), key=lambda x: results[x[1]]["f1"])[0]
    for j in range(len(col_labels)):
        table[(best_idx + 1, j)].set_facecolor("#90EE90")

    plt.tight_layout()

    if save_path:
        # Create directory if it doesn't exist
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_embeddings_tsne(
    embeddings_dict: dict[str, np.ndarray],
    labels_dict: dict[str, np.ndarray],
    save_path: Path | str | None = None,
):
    """
    Plot t-SNE visualization of embeddings.
    Args:
        embeddings_dict: {'model_name': embeddings_array, ...}
        labels_dict: {'model_name': labels_array, ...}
        save_path: Path to save figure (None = show only)
    """
    from sklearn.manifold import TSNE

    n_models = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    for idx, (name, embeddings) in enumerate(embeddings_dict.items()):
        labels = labels_dict[name]

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Plot
        ax = axes[idx]
        ax.scatter(
            embeddings_2d[labels == 0, 0],
            embeddings_2d[labels == 0, 1],
            alpha=0.6,
            label="Different authors",
            s=20,
            color="#e74c3c",
        )
        ax.scatter(
            embeddings_2d[labels == 1, 0],
            embeddings_2d[labels == 1, 1],
            alpha=0.6,
            label="Same author",
            s=20,
            color="#3498db",
        )
        ax.set_title(f"{name} - t-SNE")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        # Create directory if it doesn't exist
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    plt.show()
