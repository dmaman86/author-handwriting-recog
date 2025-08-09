import tensorflow as tf
import numpy as np
from scipy.stats import mode
from tqdm import tqdm
from typing import Callable

from utils.utils_files import get_patches

def get_predictions_confidence(
    model: tf.keras.models.Model,
    generator: tf.keras.utils.Sequence
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
    Predicts with the given model using a generator and returns predictions,
    true labels, and confidence scores.
    Args:
      model (tf.keras.models.Model): Trained classification model
      generator (tf.keras.utils.Sequence): Data generator used for prediction
    Returns:
      tuple[np.ndarray, np.ndarray, np.ndarray]:
        - y_pred: predicted class indices
        - y_true: true class indices
        - confidences: confidence of the predicted class for each sample
  """
  y_pred_probs = model.predict(generator, verbose=1)
  y_pred = np.argmax(y_pred_probs, axis=1)
  confidences = np.max(y_pred_probs, axis=1)  # probability of predicted class

  y_true = []
  for i in range(len(generator)):
    _, batch_labels = generator[i]
    y_true.extend(np.argmax(batch_labels, axis=1))
  y_true = np.array(y_true)

  return y_pred, y_true, confidences

def majority_vote_strategy(preds: np.ndarray) -> tuple[int, float]:
  '''
  Classifies using majority voting among predictions and returns confidence
  Args:
    - preds (np.ndarray): Array of shape (N, num_classes) with softmax scores
  Returns:
    - voted_class (int): Predicted class index
    - confidence (float): Confidence of the predicted class
  '''
  pred_classes = np.argmax(preds, axis=1)
  voted_class = int(mode(pred_classes, keepdims=True).mode[0])
  confidence = np.sum(pred_classes == voted_class) / len(pred_classes)

  return voted_class, confidence

def softmax_sum_strategy(preds: np.ndarray) -> tuple[int, float]:
  '''
  Classifies by summing softmax outputs across patches and selecting the class
  with the highest total.
  Args:
    - preds (np.ndarray): Array of shape (N, num_classes) with softmax scores
  Returns:
    - voted_class (int): Predicted class index (with highest summed probability)
    - confidence (float): Confidence of the predicted class
  '''
  softmax_sum = np.sum(preds, axis=0)
  voted_class = int(np.argmax(softmax_sum))
  confidence = softmax_sum[voted_class] / np.sum(softmax_sum)
  return voted_class, confidence

from typing import Callable

def evaluate_by_class(
    model: tf.keras.Model,
    test_partition: dict[int, list[str]],
    binary_dir: str,
    vote_fn: Callable[[np.ndarray], tuple[int, float]],
    dim: tuple[int, int, int] = (60, 53, 1)
) -> tuple[list[int], list[int], list[float]]:
  """
    Evaluate model predictions grouped by class_id using a provided voting function.

    Args:
        model: Trained model.
        test_partition: dict[class_id, list[patch_ids]]
        binary_dir: directory of .npy files
        vote_fn: function that takes preds (N, num_classes) and returns (voted_class, confidence)
        dim: input shape

    Returns:
        y_pred, y_true, confidences
  """

  y_pred: list[int] = []
  y_true: list[int] = []
  confidences: list[float] = []

  target_size = dim[:2]   # (height, width)

  for class_id, patch_ids in tqdm(test_partition.items(), desc="Evaluating"):
    patches = get_patches(patch_ids, binary_dir, target_size)
    y_true.append(class_id)

    X = np.array(patches, dtype=np.float32)
    preds = model.predict(X, verbose=0)

    voted_class, confidence = vote_fn(preds)
    y_pred.append(voted_class)
    confidences.append(confidence)

  return y_pred, y_true, confidences