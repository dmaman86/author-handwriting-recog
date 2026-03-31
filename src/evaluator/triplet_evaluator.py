from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from ..generators import TripletGenerator
from ..io import LoggerFactory

TripletIdxs = tuple[tf.Tensor, tf.Tensor, tf.Tensor]
TripletLabels = tuple[tf.Tensor, tf.Tensor, tf.Tensor]
TripletImages = tuple[tf.Tensor, tf.Tensor, tf.Tensor]


@dataclass
class TripletLocalMetrics:
    triplet_accuracy: float
    mean_gap: float
    triplet_error_rate: float
    auc: float


@dataclass
class TripletGlobalMetrics:
    df_acc: pd.DataFrame
    df_by_count: pd.DataFrame
    df_by_acc: pd.DataFrame
    df_by_gap: pd.DataFrame
    df_by_score: pd.DataFrame


@dataclass
class TripletReport:
    local_metrics: TripletLocalMetrics
    global_metrics: TripletGlobalMetrics
    n_triplets: int
    df_triplets: pd.DataFrame


class TripletEvaluator:

    def __init__(
        self,
        siamese_network: tf.keras.Model,
        logger: LoggerFactory,
        generator: TripletGenerator | None = None,
    ) -> None:

        self.siamese_network = siamese_network
        self.data_generator = generator

        self.logger = logger

        self._predict_triplet = tf.function(
            lambda a, p, n: self.siamese_network((a, p, n), training=False),
            reduce_retracing=True,
        )

    def _compute_embeddings(
        self,
        anchor: tf.Tensor,
        positive: tf.Tensor,
        negative: tf.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        emb_a, emb_p, emb_n = self._predict_triplet(anchor, positive, negative)

        return emb_a.numpy(), emb_p.numpy(), emb_n.numpy()

    def _compute_similarities(
        self,
        a: np.ndarray,
        p: np.ndarray,
        n: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        sim_ap = np.sum(a * p, axis=1)
        sim_an = np.sum(a * n, axis=1)

        return sim_ap, sim_an

    def _collect_triplet_results(
        self, dataset: tf.data.Dataset, steps: int
    ) -> pd.DataFrame:
        results = []

        self.logger.info("=== Collecting triplet results ===")

        for (anchor, positive, negative), meta in dataset.take(steps):
            (idx, pos_idx, neg_idx), (la, lp, ln) = meta

            a, p, n = self._compute_embeddings(anchor, positive, negative)

            sim_ap = np.sum(a * p, axis=1)
            sim_an = np.sum(a * n, axis=1)

            gap = sim_ap - sim_an
            correct = sim_ap > sim_an

            chosen_labels = np.where(correct, lp.numpy(), ln.numpy())
            chosen_idxs = np.where(correct, pos_idx.numpy(), neg_idx.numpy())

            batch_df = pd.DataFrame(
                {
                    "anchor_idx": idx.numpy(),
                    "anchor_label": la.numpy(),
                    "pred_idx": chosen_idxs,
                    "pred_label": chosen_labels,
                    "positive_label": lp.numpy(),
                    "negative_label": ln.numpy(),
                    "sim_ap": sim_ap,
                    "sim_an": sim_an,
                    "gap": gap,
                    "correct": correct,
                }
            )
            results.append(batch_df)

        self.logger.info(f"Total triplets collected: {len(results)}")

        return pd.concat(results, ignore_index=True)

    def _compute_local_metrics(self, df: pd.DataFrame) -> TripletLocalMetrics:
        triplet_accuracy = float(df["correct"].mean())
        mean_gap = float(df["gap"].mean())
        triplet_error_rate = float((df["gap"] < 0).mean())

        y_true = np.concatenate(
            [
                np.ones(len(df)),
                np.zeros(len(df)),
            ]
        )

        y_pred = np.concatenate(
            [
                df["sim_ap"].values,
                df["sim_an"].values,
            ]
        )

        auc = float(roc_auc_score(y_true, y_pred))

        self.logger.info(
            f"[LOCAL] acc={triplet_accuracy:.4f} "
            f"gap={mean_gap:.4f} "
            f"error={triplet_error_rate:.4f} "
            f"auc={auc:.4f}"
        )

        return TripletLocalMetrics(
            triplet_accuracy=triplet_accuracy,
            mean_gap=mean_gap,
            triplet_error_rate=triplet_error_rate,
            auc=auc,
        )

    def _compute_global_metrics(self, df: pd.DataFrame) -> TripletGlobalMetrics:
        self.logger.info("=== Computing global metrics ===")

        df_acc = (
            df.groupby(["anchor_label", "pred_label"])
            .agg(
                accuracy=("correct", "mean"),
                mean_gap=("gap", "mean"),
                count=("gap", "size"),
            )
            .reset_index()
        )

        gap_min = df_acc["mean_gap"].min()
        gap_max = df_acc["mean_gap"].max()
        df_acc["gap_norm"] = (df_acc["mean_gap"] - gap_min) / (gap_max - gap_min + 1e-8)
        df_acc["score"] = df_acc["accuracy"] * df_acc["gap_norm"]

        def select_best(df: pd.DataFrame, metric: str) -> pd.DataFrame:
            return (
                df.sort_values(by=metric, ascending=False)
                .groupby("anchor_label", as_index=False)
                .first()
            )

        df_by_count = select_best(df_acc, "count")
        df_by_acc = select_best(df_acc, "accuracy")
        df_by_gap = select_best(df_acc, "mean_gap")
        df_by_score = select_best(df_acc, "score")

        self.logger.info("=== Global metrics computed ===")

        return TripletGlobalMetrics(
            df_acc=df_acc,
            df_by_count=df_by_count,
            df_by_acc=df_by_acc,
            df_by_gap=df_by_gap,
            df_by_score=df_by_score,
        )

    def evaluate(self) -> TripletReport:

        if self.data_generator is None:
            raise ValueError("Data generator is not provided for evaluation.")

        self.logger.info("=== Triplet Evaluation Started ===")

        dataset = self.data_generator.build()
        steps = self.data_generator.steps_per_epoch
        df = self._collect_triplet_results(dataset, steps)

        local_metrics = self._compute_local_metrics(df)
        global_metrics = self._compute_global_metrics(df)

        self.logger.info("=== Triplet Evaluation Finished ===")

        return TripletReport(
            local_metrics=local_metrics,
            global_metrics=global_metrics,
            n_triplets=len(df),
            df_triplets=df,
        )
