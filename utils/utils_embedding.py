import tensorflow as tf
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from scipy.stats import mode
from tqdm import tqdm



def get_embeddings_from_generator(
    model: tf.keras.Model,
    generator: tf.keras.utils.Sequence
) -> tuple[np.ndarray, np.ndarray]:
    """
    Efficiently extract embeddings and true labels from a generator using Keras predict.
    
    Returns:
        embeddings: np.ndarray of shape (n_samples, embedding_dim)
        y_true: np.ndarray of shape (n_samples,)
    """
    # Predict all embeddings in batch
    embeddings = model.predict(generator, verbose=1)

    # Extract true labels
    y_true = np.array([
        np.argmax(generator.labels[patch_id]) if isinstance(generator.labels[patch_id], (list, np.ndarray)) 
        else generator.labels[patch_id]
        for patch_id in generator.list_ids
    ])

    return embeddings, y_true

def build_gallery(embeddings: np.ndarray, labels: np.ndarray) -> dict[int, list[np.ndarray]]:
  '''
  Groups embeddings by class label into a gallery dictionary.
  Args:
    - embeddings (np.ndarray): Array of embeddings.
    - labels (np.ndarray): Array of class labels.
  Returns:
    dict[int, list[np.ndarray]]: Mapping from class ID to list of embeddings.
  '''
  gallery = defaultdict(list)
  for emb, label in zip(embeddings, labels):
    gallery[label].append(emb)
  return gallery

'''
def compute_class_centroids(gallery: dict[int, list[np.ndarray]]) -> dict[int, np.ndarray]:
  Computes a centroid (mean vector) for each class based on its embeddings.
  Args:
    - gallery (dict[int, list[np.ndarray]]): Dictionary of embeddings per class.
  Returns:
    dict[int, np.ndarray]: Mapping from class ID to centroid.
  return {cid: np.mean(np.stack(embs), axis=0) for cid, embs in gallery.items()}
'''
def compute_class_centroids(gallery: dict[int, list[np.ndarray]]) -> dict[int, np.ndarray]:
  """
  Compute L2-normalized centroids per class from a gallery of embeddings.
  """
  centroids: dict[int, np.ndarray] = {}
  for cid, vecs in gallery.items():
    M = np.stack(vecs, axis=0) # Shape: (n_samples, embedding_dim)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)  # L2-normalize rows
    c = M.mean(axis=0)
    c = c / (np.linalg.norm(c) + 1e-12)
    centroids[cid] = c.astype(np.float32)  # Ensure float32 type
  return centroids

def predict_by_centroid_distance(
    test_embeddings: np.ndarray,
    centroids: dict[int, np.ndarray]
) -> tuple[list[int], list[float]]:
  '''
  Predicts the class of each embedding by finding the closest centroid.
  Args:
    - test_embeddings (np.ndarray): Array of embeddings for test set.
    - centroids (dict[int, np.ndarray]): Mapping from class ID to centroid.
  Returns:
    tuple[list[int], list[float]]: List of predicted classes and corresponding 
    confidence scores.
  '''
  class_ids = list(centroids.keys())
  centroid_matrix = np.stack([centroids[cid] for cid in class_ids])
  y_pred = []
  confidences = []

  for emb in test_embeddings:
    distances = cdist(np.expand_dims(emb, axis=0), centroid_matrix, metric="euclidean")[0]
    best_idx = int(np.argmin(distances))
    predicted_class = class_ids[best_idx]
    confidence = 1 / (1 + distances[best_idx])
    y_pred.append(predicted_class)
    confidences.append(confidence)

  return y_pred, confidences

def get_patch_embeddings_from_generator(generator: tf.keras.utils.Sequence, model: tf.keras.Model) -> dict[str, np.ndarray]:
    """
    Compute embeddings for all patches in a generator using batched prediction.

    Returns:
        A dictionary mapping patch_id to embedding vector.
    """
    raw_embeddings = model.predict(generator, verbose=1)
    return {
        patch_id: embedding
        for patch_id, embedding in zip(generator.list_ids, raw_embeddings)
    }

def vote_by_majority_emb(embeddings: list[np.ndarray], centroids: dict[int, np.ndarray]) -> tuple[int, float]:
  '''
  Predicts the class of a list of embeddings by majority voting.
  Args:
    - embeddings (list[np.ndarray]): List of embeddings to classify.
    - centroids (dict[int, np.ndarray]): Mapping from class ID to centroid.
  Returns:
    - Predicted class (via majority vote)
    - Confidence score = proportion of embeddings that voted for the predicted 
    class  
  '''
  class_ids = list(centroids.keys())
  centroid_matrix = np.stack([centroids[cid] for cid in class_ids])
  predictions = []

  for emb in embeddings:
    dists = cdist([emb], centroid_matrix, metric="euclidean")[0]
    best_idx = int(np.argmin(dists))
    predictions.append(class_ids[best_idx])

  voted_class = int(mode(predictions, keepdims=True).mode[0])
  confidence = np.sum(np.array(predictions) == voted_class) / len(predictions)
  return voted_class, confidence

def vote_by_softmax_sum_emb(embeddings: list[np.ndarray], centroids: dict[int, np.ndarray]) -> tuple[int, float]:
  '''
  Predicts class by summing softmax probabilities over distances to centroids.
  Args:
    - embeddings (list[np.ndarray]): List of embeddings to classify.
    - centroids (dict[int, np.ndarray]): Mapping from class ID to centroid.
  Returns:
    - Predicted class (with highest accumulated softmax)
    - Confidence score
  '''
  class_ids = list(centroids.keys())
  centroid_matrix = np.stack([centroids[cid] for cid in class_ids])
  total_softmax = np.zeros(len(class_ids))

  for emb in embeddings:
    distances = cdist([emb], centroid_matrix, metric="euclidean")[0]
    logits = -distances
    softmax = np.exp(logits) / np.sum(np.exp(logits))
    total_softmax += softmax

  best_idx = int(np.argmax(total_softmax))
  predicted_class = class_ids[best_idx]
  confidence = total_softmax[best_idx] / np.sum(total_softmax)
  return predicted_class, confidence

def evaluate_by_group_emb(
    class_to_patch_ids: dict[int, list[str]],
    patch_id_to_embedding: dict[str, np.ndarray],
    centroids: dict[int, np.ndarray],
    vote_fn
) -> tuple[list[int], list[int], list[float]]:
  '''
  Evaluates grouped patch predictions using a voting function over embeddings.
  Args:
    - class_to_patch_ids (dict[int, list[str]]): Mapping from class ID to list of patch IDs.
    - patch_id_to_embedding (dict[str, np.ndarray]): Patch ID -> embedding vector
    - centroids (dict[int, np.ndarray]): Mapping from class ID to centroid.
    - vote_fn: Function to perform voting over embeddings.
  Returns:
    - y_pred: predicted class per group
    - y_true: true class per group
    - confidences: confidence scores per prediction
  '''
  y_true = []
  y_pred = []
  confidences = []

  for class_id, patch_ids in tqdm(class_to_patch_ids.items(), desc="Evaluating by group", total=len(class_to_patch_ids)):
    embeddings = [patch_id_to_embedding[pid] for pid in patch_ids]
    pred_class, confidence = vote_fn(embeddings, centroids)

    y_true.append(class_id)
    y_pred.append(pred_class)
    confidences.append(confidence)

  return y_pred, y_true, confidences
