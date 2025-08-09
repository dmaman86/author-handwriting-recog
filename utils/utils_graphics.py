import numpy as np
import visualkeras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
import umap.umap_ as umap


def plot_metric(ax: plt.Axes, train_values: list[float], val_values: list[float],
                title: str, xlabel: str, ylabel: str, labels=('Train', 'Validation')) -> plt.Axes:
    """Plot train and validation values on a given Axes."""
    ax.plot(train_values)
    ax.plot(val_values)
    ax.title.set_text(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(labels, loc='upper left')
    return ax

def dplot_graph(accuracy: list[float], val_accuracy: list[float], loss: list[float], val_loss: list[float]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    plot_metric(ax1, accuracy, val_accuracy, 'Accuracy Model', 'Epochs', 'Accuracy')
    plot_metric(ax2, loss, val_loss, 'Loss Model', 'Epochs', 'Loss')

    fig.tight_layout()
    plt.show()


def plot_activations(
    activations: list[np.ndarray],
    layer_names: list[str],
    images_per_row: int = 8,
    grid_cell_size: tuple[float, float] = (1.5, 1.2)
) -> None:
    """
    Visualize activation maps for a list of layers with custom grid size.

    Args:
        activations (list[np.ndarray]): List of activations from intermediate layers.
        layer_names (list[str]): Corresponding layer names.
        images_per_row (int): Number of feature maps to show per row.
        grid_cell_size (tuple[float, float]): Size (width, height) per grid cell in inches.
    """
    for layer_name, layer_activation in zip(layer_names, activations):
      n_features = layer_activation.shape[-1]
      height = layer_activation.shape[1]
      width = layer_activation.shape[2]

      n_cols = max(1, n_features // images_per_row)

      display_grid = np.zeros((height * n_cols, width * images_per_row))

      for col in range(n_cols):
        for row in range(images_per_row):
          idx = col * images_per_row + row
          if idx >= n_features: break

          channel_image = layer_activation[0, :, :, idx]
          channel_image -= channel_image.mean()
          channel_image /= (channel_image.std() + 1e-5)
          channel_image *= 64
          channel_image += 128
          channel_image = np.clip(channel_image, 0, 255).astype("uint8")

          display_grid[
            col * height : (col + 1) * height,
            row * width : (row + 1) * width
          ] = channel_image

      fig_width = images_per_row * grid_cell_size[0]
      fig_height = n_cols * grid_cell_size[1]
      plt.figure(figsize=(fig_width, fig_height))
      plt.title(layer_name)
      plt.grid(False)
      plt.imshow(display_grid, cmap='gray', interpolation='antialiased')
    plt.show()

def plot_all_confusion_matrices(results: dict[str, dict], figsize: tuple[int, int] = (20, 20)) -> None:
  """
  Plots multiple confusion matrices (e.g., one for each evaluation strategy)
  Args:
    - results (dict[str, dict]): Dictionary with keys as strategy names and 
    values containing:
      - 'data': tuple of (y_pred, y_true)
      - 'title': title of the confusion matrix
    - figsize (tuple[int, int]): Size of the figure.
  """
  fig, axes = plt.subplots(2, 2, figsize=figsize)

  for ax, (key, result_info) in zip(axes.flat, results.items()):
    y_pred, y_true = result_info["data"]
    title_base = result_info["title"]
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    cax1 = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax1, ax=ax)
    ax.set_title(f"{title_base} (Acc: {acc:.2%})", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

  plt.tight_layout()
  plt.show()

def plot_embedding_subset(
    embeddings_dict: dict[str, np.ndarray],
    labels_dict: dict[str, int],
    reducer,                     # Pre-initialized t-SNE or UMAP reducer
    num_classes: int = 12,
    random_state: int = 42,
    title: str = "Embedding Projection (subset)"
) -> None:
    """
    Visualize a 2D projection of a random subset of classes using t-SNE or UMAP.

    Args:
        embeddings_dict (dict[str, np.ndarray]): Mapping from patch ID to its embedding vector.
        labels_dict (dict[str, int]): Mapping from patch ID to its class label.
        reducer: An instance of a dimensionality reduction class (e.g., UMAP or TSNE).
        num_classes (int): Number of classes to randomly select for visualization.
        random_state (int): Random seed for reproducibility.
        title (str): Plot title.
    """
    # Extract embeddings and labels
    patch_ids = list(embeddings_dict.keys())
    embeddings = np.array([embeddings_dict[pid] for pid in patch_ids])
    labels = np.array([labels_dict[pid] for pid in patch_ids])

    # Select a random subset of class labels
    all_classes = np.unique(labels)
    rng = np.random.default_rng(seed=random_state)
    selected_classes = rng.choice(all_classes, size=num_classes, replace=False)

    # Filter data to include only selected classes
    mask = np.isin(labels, selected_classes)
    X_subset = embeddings[mask]
    y_subset = labels[mask]

    # Perform dimensionality reduction
    X_reduced = reducer.fit_transform(X_subset)

    # Plot the result
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_subset, cmap='tab20', s=10)
    plt.colorbar(scatter)
    plt.title(f"{title} - {num_classes} Classes")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()