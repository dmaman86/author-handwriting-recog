import numpy as np

from .base_identification_evaluator import BaseIdentificationEvaluator


class UnseenIdentificationEvaluator(BaseIdentificationEvaluator):

    def _get_mode_name(self) -> str:
        return "open-set"

    def _evaluate_identification(
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
