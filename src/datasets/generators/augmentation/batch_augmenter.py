import numpy as np
import tensorflow as tf


class BatchAugmenter:

    def __init__(self, augmentation_pipeline: tf.keras.Sequential) -> None:
        self._pipeline = augmentation_pipeline

    def augment_pairs(
        self, X1: np.ndarray, X2: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        X1_aug = []
        X2_aug = []

        positive_mask = y == 1
        if np.any(positive_mask):
            X1_pos, X2_pos = self._augment_positive_pairs(
                X1[positive_mask], X2[positive_mask]
            )
            X1_aug.append(X1_pos)
            X2_aug.append(X2_pos)

        negative_mask = y == 0
        if np.any(negative_mask):
            X1_neg, X2_neg = self._augment_negative_pairs(
                X1[negative_mask], X2[negative_mask]
            )
            X1_aug.append(X1_neg)
            X2_aug.append(X2_neg)

        return self._reconstruct_batch_order(X1_aug, X2_aug, X1, X2, positive_mask)

    def _augment_positive_pairs(
        self,
        X1_pos: np.ndarray,
        X2_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_pos_concat = np.concatenate([X1_pos, X2_pos], axis=0)
        X_pos_aug = self._pipeline(X_pos_concat, training=True).numpy()

        # split back
        n_pos = len(X1_pos)
        X1_pos_aug = X_pos_aug[:n_pos]
        X2_pos_aug = X_pos_aug[n_pos:]

        return X1_pos_aug, X2_pos_aug

    def _augment_negative_pairs(
        self,
        X1_neg: np.ndarray,
        X2_neg: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X1_neg_aug = self._pipeline(X1_neg, training=True).numpy()
        X2_neg_aug = self._pipeline(X2_neg, training=True).numpy()

        return X1_neg_aug, X2_neg_aug

    def _reconstruct_batch_order(
        self,
        X1_aug: list[np.ndarray],
        X2_aug: list[np.ndarray],
        X1_original: np.ndarray,
        X2_original: np.ndarray,
        positive_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X1_final = np.zeros_like(X1_original)
        X2_final = np.zeros_like(X2_original)

        if len(X1_aug) == 0:
            return X1_final, X2_final

        pos_idx, neg_idx = 0, 0

        for i, is_positive in enumerate(positive_mask):
            if is_positive:
                X1_final[i] = (
                    X1_aug[0][pos_idx]
                    if len(X1_aug) > 0 and pos_idx < len(X1_aug[0])
                    else X1_original[i]
                )
                X2_final[i] = (
                    X2_aug[0][pos_idx]
                    if len(X2_aug) > 0 and pos_idx < len(X2_aug[0])
                    else X2_original[i]
                )
                pos_idx += 1
            else:
                aug_idx = 1 if len(X1_aug) > 1 else 0
                X1_final[i] = (
                    X1_aug[aug_idx][neg_idx]
                    if aug_idx < len(X1_aug) and neg_idx < len(X1_aug[aug_idx])
                    else X1_original[i]
                )
                X2_final[i] = (
                    X2_aug[aug_idx][neg_idx]
                    if aug_idx < len(X2_aug) and neg_idx < len(X2_aug[aug_idx])
                    else X2_original[i]
                )
                neg_idx += 1

        return X1_final, X2_final
