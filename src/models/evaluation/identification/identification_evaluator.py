from typing import Literal

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from ....io.logging import LoggerFactory
from .gallery.gallery_builder import GalleryBuilder
from .strategy.base import IdentificationStrategy

ModelType = Literal["open-set", "close-set"]


class IdentificationEvaluator:
    """
    Closed-set writer identification evaluator based on embeddings.
    """

    def __init__(
        self,
        embedding_model: tf.keras.Model,
        strategy: IdentificationStrategy,
        gallery_builder: GalleryBuilder,
        mode: ModelType = "close-set",
        gallery_ratio: float = 0.5,
        min_probe: int = 2,
        seed: int = 42,
        log_dir: str | None = None,
        model_name: str = "embedding_model",
    ) -> None:
        self.embedding_model = embedding_model
        self.strategy = strategy
        self.gallery_builder = gallery_builder
        self.mode = mode
        self.gallery_ratio = gallery_ratio
        self.min_probe = min_probe
        self.seed = seed

        self.logger = LoggerFactory.get_logger(
            name=f"{model_name}_identification",
            log_dir=log_dir,
            file_prefix=f"{model_name}_identification",
        )

    def evaluate(self, dataset: tf.data.Dataset, steps: int) -> dict:
        if steps <= 0:
            raise ValueError(f"Steps must be > 0, got {steps}")

        self.logger.info("Starting identification evaluation")
        self.logger.info(f"  Strategy: {self.strategy.__class__.__name__}")
        self.logger.info(f"  Gallery ratio: {self.gallery_ratio}")
        self.logger.info(f"  Min probe samples: {self.min_probe}")

        embeddings, author_ids = self._extract_embeddings(dataset, steps)

        gallery, probe_embeddings, probe_author_ids = self.gallery_builder.build(
            embeddings=embeddings,
            author_ids=author_ids,
            gallery_ratio=self.gallery_ratio,
            seed=self.seed,
            min_probe=self.min_probe,
        )

        if self.mode == "close-set":
            return self._evaluate_closedset(gallery, probe_embeddings, probe_author_ids)
        else:
            return self._evaluate_openset(gallery, probe_embeddings, probe_author_ids)

    def _evaluate_closedset(
        self,
        gallery: dict[int, np.ndarray],
        probe_embeddings: np.ndarray,
        probe_author_ids: np.ndarray,
    ) -> dict:
        y_true, y_pred, details = [], [], []

        for author_id in np.unique(probe_author_ids):
            idx = np.where(probe_author_ids == author_id)[0]
            if len(idx) < self.min_probe:
                continue

            res = self.strategy.identify(probe_embeddings[idx], gallery)
            y_true.append(int(author_id))
            y_pred.append(res["pred_author_id"])
            details.append(
                {
                    "true_author_id": int(author_id),
                    **res,
                }
            )
        accuracy = accuracy_score(y_true, y_pred)

        self.logger.info("Closed-set identification evaluation completed")
        self.logger.info(f"  Num authors evaluated: {len(y_true)}")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")

        return {
            "accuracy": accuracy,
            "num_authors": len(y_true),
            "y_true": np.array(y_true),
            "y_pred": np.array(y_pred),
            "details": details,
        }

    def _evaluate_openset(
        self,
        gallery: dict[int, np.ndarray],
        probe_embeddings: np.ndarray,
        probe_author_ids: np.ndarray,
    ) -> dict:
        known_author_ids = set(gallery.keys())

        y_true, y_pred, details = [], [], []
        y_true_binary = []  # 1=known, 0=unknown
        y_pred_binary = []  # 1=identified, 0=rejected

        for author_id in np.unique(probe_author_ids):
            idx = np.where(probe_author_ids == author_id)[0]
            if len(idx) < self.min_probe:
                continue
            res = self.strategy.identify(probe_embeddings[idx], gallery)
            is_known = int(author_id) in known_author_ids
            y_true_binary.append(1 if is_known else 0)

            pred_id = res["pred_author_id"]
            y_pred_binary.append(1 if pred_id is not None else 0)

            true_label = int(author_id) if is_known else -1
            pred_label = pred_id if pred_id is not None else -1

            y_true.append(true_label)
            y_pred.append(pred_label)
            details.append(
                {
                    "true_author_id": int(author_id),
                    "is_known": is_known,
                    "was_rejected": pred_id is None,
                    **res,
                }
            )

        metrics = self._compute_openset_metrics(
            np.array(y_true_binary),
            np.array(y_pred_binary),
            np.array(y_true),
            np.array(y_pred),
        )
        self.logger.info("Open-set identification evaluation completed")
        self.logger.info(f"  Num known authors: {metrics['num_known']}")
        self.logger.info(f"  Num unknown authors: {metrics['num_unknown']}")
        self.logger.info(f"  Detection rate: {metrics['detection_rate']:.4f}")
        self.logger.info(f"  False alarm rate: {metrics['false_alarm_rate']:.4f}")
        self.logger.info(f"  Rejection rate: {metrics['rejection_rate']:.4f}")

        return {
            **metrics,
            "y_true": np.array(y_true),
            "y_pred": np.array(y_pred),
            "y_true_binary": np.array(y_true_binary),
            "y_pred_binary": np.array(y_pred_binary),
            "details": details,
        }

    def _compute_openset_metrics(
        self,
        y_true_binary: np.ndarray,
        y_pred_binary: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, any]:
        known_mask = y_true_binary == 1
        unknown_mask = y_true_binary == 0

        num_known = int(known_mask.sum())
        num_unknown = int(unknown_mask.sum())

        if num_known > 0:
            known_correct = (y_true[known_mask] == y_pred[known_mask]).sum()
            detection_rate = float(known_correct / num_known)
        else:
            detection_rate = 0.0

        if num_unknown > 0:
            false_alarms = (y_pred_binary[unknown_mask] == 1).sum()
            false_alarm_rate = float(false_alarms / num_unknown)
        else:
            false_alarm_rate = 0.0

        rejection_rate = 1.0 - false_alarm_rate if num_unknown > 0 else 0.0
        closed_set_accuracy = detection_rate

        return {
            "detection_rate": detection_rate,
            "false_alarm_rate": false_alarm_rate,
            "rejection_rate": rejection_rate,
            "closed_set_accuracy": closed_set_accuracy,
            "num_known": num_known,
            "num_unknown": num_unknown,
        }

    def _extract_embeddings(
        self, dataset: tf.data.Dataset, steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        self.logger.info(f"Extracting embeddings from {steps} batches...")

        all_embeddings, all_author_ids = [], []
        log_interval = max(1, steps // 10)

        for i, (X, y) in enumerate(dataset.take(steps)):
            if i % log_interval == 0:
                self.logger.info(f"  Processing batch {i + 1}/{steps}...")

            emb = self.embedding_model(X, training=False)
            all_embeddings.append(emb.numpy())

            if isinstance(y, np.ndarray):
                all_author_ids.append(y)
            else:
                all_author_ids.append(y.numpy())

        embeddings = np.concatenate(all_embeddings, axis=0)
        author_ids = np.concatenate(all_author_ids, axis=0)

        self.logger.info(
            f"Extracted {len(embeddings)} embeddings "
            f"from {len(np.unique(author_ids))} unique authors"
        )

        return embeddings, author_ids
