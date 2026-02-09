from pathlib import Path
from typing import Literal

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)

from ....io.logging import LoggerFactory
from .thresholding import (ThresholdFinder, ThresholdFinderFactory,
                           ThresholdStrategy)

ModelType = Literal["pair", "triplet"]
GeneratorType = tf.data.Dataset | tf.keras.utils.Sequence


class VerificationEvaluator:

    def __init__(
        self,
        model: tf.keras.Model,
        model_name: str,
        model_type: ModelType = "pair",
        threshold_finder: ThresholdFinder | None = None,
        log_dir: Path | str | None = None,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.model_type = model_type
        self.threshold_finder = threshold_finder

        self.logger = LoggerFactory.get_logger(
            name=f"{model_name}_evaluator",
            log_dir=log_dir,
            file_prefix=f"{model_name}_eval",
        )

    def evaluate(
        self,
        generator: GeneratorType,
        steps: int | None = None,
        distance_threshold: float | None = None,
        threshold_strategy: ThresholdStrategy = "youden",
    ) -> dict:
        distances, y_true = self._compute_distances(generator, steps)

        if distance_threshold is None:
            finder = (
                self.threshold_finder
                if self.threshold_finder
                else ThresholdFinderFactory.create(threshold_strategy)
            )
            distance_threshold = finder.find(y_true, distances)

        results = self._compute_metrics(y_true, distances, distance_threshold)

        self.logger.info("Verification evaluation completed")
        self.logger.info(f"  Accuracy:  {results['accuracy']:.4f}")
        self.logger.info(f"  AUC:       {results['auc']:.4f}")
        self.logger.info(f"  Precision: {results['precision']:.4f}")
        self.logger.info(f"  Recall:    {results['recall']:.4f}")
        self.logger.info(f"  F1-score:  {results['f1']:.4f}")

        return results

    def _compute_distances(self, generator: GeneratorType, steps: int | None):
        distances, y_true = [], []

        dataset = self._get_dataset(generator, steps)

        if self.model_type == "pair":
            distances, y_true = self._evaluate_pairs(dataset)
        elif self.model_type == "triplet":
            distances, y_true = self._evaluate_triplets(dataset)

        return np.concatenate(distances), np.concatenate(y_true)

    def _get_dataset(self, generator: GeneratorType, steps: int | None):
        if isinstance(generator, tf.data.Dataset):
            if steps is None:
                card = tf.data.experimental.cardinality(generator).numpy()
                if card > 0:
                    steps = card
                else:
                    raise ValueError(
                        "Steps must be provided for tf.data.Dataset evaluation"
                    )
            return generator.take(steps)
        else:
            return (generator[i] for i in range(len(generator)))

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

    def _compute_metrics(self, y_true, distances, threshold):
        y_pred = (distances < threshold).astype(int)
        scores = -distances  # proper metric-learning score

        return {
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )[0],
            "recall": precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )[1],
            "f1": precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )[2],
            "auc": roc_auc_score(y_true, scores),
            "distances": distances,
            "y_true": y_true,
            "y_pred": y_pred,
        }
