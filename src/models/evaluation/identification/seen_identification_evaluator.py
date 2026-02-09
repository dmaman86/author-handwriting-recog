import numpy as np
from sklearn.metrics import accuracy_score

from .base_identification_evaluator import BaseIdentificationEvaluator


class SeenIdentificationEvaluator(BaseIdentificationEvaluator):

    def _get_mode_name(self) -> str:
        return "closed-set"

    def _evaluate_identification(
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
