"""
Evaluator for individual and ensemble Siamese models.
"""

from pathlib import Path
from typing import Any, Literal

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

from ...io.logging import LoggerFactory

ModelType = Literal["pair", "triplet"]
GeneratorType = tf.data.Dataset | tf.keras.utils.Sequence
ThresholdStrategy = Literal["youden", "f1", "eer"]


class ModelEvaluator:
    """
    Evaluate individual Siamese model.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        model_name: str,
        embedding_model: tf.keras.Model | None = None,
        model_type: ModelType = "pair",
        log_dir: Path | str | None = None,
    ) -> None:
        """
        Initialize evaluator.

        Args:
            model: Trained Siamese model
            model_name: Model name for logging
            log_dir: Directory for logs
        """
        self.model = model
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.model_type = model_type

        self.logger = LoggerFactory.get_logger(
            name=f"{model_name}_evaluator",
            log_dir=log_dir,
            file_prefix=f"{model_name}_eval",
        )

        self.results: dict[str, Any] = {}

    def evaluate(
        self,
        test_generator: GeneratorType,
        distance_threshold: float | None = None,
        distance_scale: float | None = None,
        steps: int | None = None,
        threshold_strategy: ThresholdStrategy = "youden",
    ) -> dict[str, float]:
        """
        Evaluate model on test generator.

        Args:
            test_generator: Test data generator (PairGenerator)
            distance_threshold: Threshold for binary classification.
                If None, the optimal threshold is found automatically
                using the chosen ``threshold_strategy``.
            distance_scale: Optional scale used to convert distances to logits.
                If None, tries to read ``scale`` attribute from the model loss
                (used by BinaryCrossEntropyDistance). Defaults to 1.0.
            steps: Number of batches to evaluate (required for infinite
                ``tf.data.Dataset`` generators).
            threshold_strategy: Strategy for finding the optimal distance
                threshold when ``distance_threshold`` is None.
                - ``"youden"``: Maximises Youden's J statistic (TPR - FPR).
                - ``"f1"``: Maximises the F1-score over a grid of thresholds.
                - ``"eer"``: Equal Error Rate (where FPR ≈ FNR).

        Returns:
            Dictionary with metrics
        """
        self.logger.info(f"Evaluating {self.model_name}...")

        # Extract data from generator
        y_true_all: list[np.ndarray] = []
        distances_all: list[np.ndarray] = []

        if isinstance(test_generator, tf.data.Dataset):
            if steps is None:
                card = tf.data.experimental.cardinality(test_generator).numpy()
                if card > 0:
                    steps = card
                else:
                    raise ValueError(
                        "Steps must be provided for tf.data.Dataset evaluation"
                    )
            dataset = test_generator.take(steps)
        else:
            dataset = (test_generator[i] for i in range(len(test_generator)))

        # Try to infer distance scale from loss if not provided
        if distance_scale is None:
            loss = getattr(self.model, "loss", None)
            distance_scale = float(getattr(loss, "scale", 1.0)) if loss else 1.0

        if self.model_type == "pair":
            distances_all, y_true_all = self._evaluate_pairs(dataset)
        elif self.model_type == "triplet":
            distances_all, y_true_all = self._evaluate_triplets(dataset)

        y_true = np.concatenate(y_true_all, axis=0)
        distances = np.concatenate(distances_all, axis=0)

        # Find optimal threshold if not provided
        if distance_threshold is None:
            distance_threshold = self._find_optimal_threshold(
                y_true, distances, strategy=threshold_strategy
            )
            self.logger.info(
                f"Optimal threshold ({threshold_strategy}): {distance_threshold:.4f}"
            )

        return self._compute_metrics(
            y_true, distances, distance_threshold, distance_scale
        )

    def _evaluate_pairs(self, dataset) -> tuple[list[np.ndarray], list[np.ndarray]]:
        distances_all, y_true_all = [], []

        for (img1, img2), labels in dataset:
            embeddings = self.model.predict([img1, img2], verbose=0)
            emb1 = embeddings[:, 0, :]
            emb2 = embeddings[:, 1, :]
            distances = np.linalg.norm(emb1 - emb2, axis=1)

            distances_all.append(distances)
            y_true_all.append(
                labels if isinstance(labels, np.ndarray) else labels.numpy()
            )

        return distances_all, y_true_all

    def _evaluate_triplets(self, dataset) -> tuple[list[np.ndarray], list[np.ndarray]]:
        distances_all, y_true_all = [], []

        for (anchor, positive, negative), _ in dataset:
            embeddings = self.model.predict([anchor, positive, negative], verbose=0)
            emb_anchor = embeddings[:, 0, :]
            emb_positive = embeddings[:, 1, :]
            emb_negative = embeddings[:, 2, :]

            pos_distances = np.sqrt(
                np.sum(np.square(emb_anchor - emb_positive), axis=1)
            )
            distances_all.append(pos_distances)
            y_true_all.append(np.ones(len(pos_distances)))

            neg_distances = np.sqrt(
                np.sum(np.square(emb_anchor - emb_negative), axis=1)
            )
            distances_all.append(neg_distances)
            y_true_all.append(np.zeros(len(neg_distances)))

        return distances_all, y_true_all

    # ------------------------------------------------------------------
    # Optimal threshold search
    # ------------------------------------------------------------------

    def _find_optimal_threshold(
        self,
        y_true: np.ndarray,
        distances: np.ndarray,
        strategy: ThresholdStrategy = "youden",
    ) -> float:
        """
        Find the optimal distance threshold for binary classification.

        Three strategies are available:

        * **youden** – Maximises *Youden's J statistic* (sensitivity + specificity − 1)
          using the ROC curve.  This is the most common choice when you want to
          balance true-positive and false-positive rates.

        * **f1** – Sweeps a grid of candidate thresholds and picks the one that
          maximises the *F1-score*.  Preferred when precision/recall balance matters
          more than raw TPR/FPR trade-off.

        * **eer** – Finds the *Equal Error Rate* point, where FPR ≈ FNR.
          Common in biometric verification tasks.

        Args:
            y_true: Ground-truth binary labels (1 = same author, 0 = different).
            distances: Euclidean distances between embedding pairs.
            strategy: One of ``"youden"``, ``"f1"``, or ``"eer"``.

        Returns:
            Optimal distance threshold (float).
        """
        if strategy == "youden":
            return self._threshold_youden(y_true, distances)
        elif strategy == "f1":
            return self._threshold_f1(y_true, distances)
        elif strategy == "eer":
            return self._threshold_eer(y_true, distances)
        else:
            raise ValueError(
                f"Unknown threshold strategy '{strategy}'. "
                "Choose from: 'youden', 'f1', 'eer'."
            )

    def _threshold_youden(self, y_true: np.ndarray, distances: np.ndarray) -> float:
        """Youden's J statistic: maximise (TPR − FPR) on the ROC curve.

        We use ``1 − distance`` as the similarity score so that
        ``roc_curve`` (which expects higher = more positive) works correctly.
        """
        similarity = -distances  # monotonic transform; higher = more similar
        fpr, tpr, thresholds = roc_curve(y_true, similarity)

        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))

        # roc_curve thresholds are in *similarity* space → convert back
        optimal_distance = -float(thresholds[best_idx])

        self.logger.info(
            f"  Youden's J = {j_scores[best_idx]:.4f} "
            f"(TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f})"
        )
        return optimal_distance

    def _threshold_f1(
        self, y_true: np.ndarray, distances: np.ndarray, n_candidates: int = 200
    ) -> float:
        """Grid-search over candidate thresholds to maximise F1-score."""
        candidates = np.linspace(
            float(distances.min()), float(distances.max()), n_candidates
        )

        best_f1 = -1.0
        best_threshold = float(candidates[0])

        for t in candidates:
            y_pred = (distances < t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)

        self.logger.info(
            f"  Best F1 = {best_f1:.4f} at threshold = {best_threshold:.4f}"
        )
        return best_threshold

    def _threshold_eer(self, y_true: np.ndarray, distances: np.ndarray) -> float:
        """Equal Error Rate: find the threshold where FPR ≈ FNR."""
        similarity = -distances
        fpr, tpr, thresholds = roc_curve(y_true, similarity)
        fnr = 1 - tpr

        # Find the crossing point where |FPR − FNR| is minimised
        eer_idx = int(np.argmin(np.abs(fpr - fnr)))
        eer_value = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

        optimal_distance = -float(thresholds[eer_idx])

        self.logger.info(f"  EER = {eer_value:.4f}")
        return optimal_distance

    # ------------------------------------------------------------------
    # Probability conversion
    # ------------------------------------------------------------------

    def _distances_to_probabilities(
        self, distances: np.ndarray, distance_scale: float
    ) -> np.ndarray:
        """Convert Euclidean distances to similarity probabilities.

        Uses ``sigmoid(-scale * distance)`` so that:
        - distance → 0  ⟹  probability → 1  (same author)
        - distance → ∞  ⟹  probability → 0  (different author)

        The ``distance_scale`` controls the sigmoid sharpness and is
        resolved once in ``evaluate()`` from the model loss or caller.

        Args:
            distances: Euclidean distances between embedding pairs.
            distance_scale: Sharpness factor for the sigmoid conversion.

        Returns:
            Array of similarity probabilities in [0, 1].
        """
        logits = -distance_scale * distances
        return 1.0 / (1.0 + np.exp(-logits))

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        distances: np.ndarray,
        threshold: float,
        distance_scale: float = 1.0,
    ) -> dict[str, float]:
        """Compute classification metrics from distances.

        Args:
            y_true: Ground-truth binary labels.
            distances: Euclidean distances between embedding pairs.
            threshold: Distance threshold for binary decision.
            distance_scale: Scale already resolved by ``evaluate()``.

        Returns:
            Dictionary with all evaluation metrics and raw arrays.
        """
        y_pred = (distances < threshold).astype(int)
        y_pred_proba = self._distances_to_probabilities(distances, distance_scale)

        accuracy = accuracy_score(y_true, y_pred)

        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except Exception:
            auc = 0.0
            self.logger.warning("Could not calculate AUC")

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        self.results = {
            "accuracy": accuracy,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "threshold": threshold,
            "distances": distances,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

        self.logger.info("Evaluation completed:")
        self.logger.info(f"  - Threshold: {threshold:.4f}")
        self.logger.info(f"  - Accuracy:  {accuracy:.4f}")
        self.logger.info(f"  - AUC:       {auc:.4f}")
        self.logger.info(f"  - Precision: {precision:.4f}")
        self.logger.info(f"  - Recall:    {recall:.4f}")
        self.logger.info(f"  - F1-Score:  {f1:.4f}")

        return self.results
