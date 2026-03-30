from dataclasses import dataclass
from typing import Callable

import pandas as pd
import tensorflow as tf

from .evaluator import (ErrorAnalysis, EvaluatedPrediction,
                        PairStrategyMetrics, PredictionResult)


@dataclass
class PairEvaluationReport:
    df_rows: pd.DataFrame
    df_pairs: pd.DataFrame
    strategy_results: dict[str, PairStrategyMetrics]


class PairEvaluator:

    def __init__(
        self,
        siamese_network: tf.keras.Model,
    ) -> None:
        self.siamese_network = siamese_network

        self.dataset: tf.data.Dataset | None = None
        self.steps: int | None = None

        self.df_rows: pd.DataFrame | None = None
        self.df_pairs: pd.DataFrame | None = None

        self.score_functions: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
            "prob": lambda df: df["prob_mean"],
            "votes": lambda df: df["vote_ratio"],
            "hybrid": lambda df: df["vote_ratio"] * df["prob_mean"],
            "sum": lambda df: df["prob_sum"],
        }

    def update_data(
        self, dataset: tf.data.Dataset, steps: int, threshold: float = 0.5
    ) -> None:
        self.dataset = dataset
        self.steps = steps

        self.df_rows = self._build_dataframe(threshold)
        self.df_pairs = self._build_pairs(self.df_rows)

    def evaluate(self) -> PairEvaluationReport:

        if self.df_rows is None or self.df_pairs is None:
            raise ValueError(
                "Data not initialized. Call update_data() before evaluation."
            )

        strategy_results: dict[str, PairStrategyMetrics] = {}

        for strategy, fn in self.score_functions.items():
            df_pairs = self.df_pairs.copy()
            df_pairs["score"] = fn(df_pairs)

            # =========================
            # PATCH LEVEL
            # =========================

            df_patch = self.df_rows.copy()
            patch_acc = (df_patch["y_true"] == df_patch["y_pred"]).mean()

            df_patch_errors = df_patch[df_patch["y_true"] != df_patch["y_pred"]]

            # =========================
            # SCORE (best match per author)
            # =========================

            df_score = (
                df_pairs.sort_values("score", ascending=False)
                .groupby("author_i")
                .head(1)
                .reset_index(drop=True)
            )

            score_acc = (df_score["author_i"] == df_score["author_j"]).mean()

            df_score_errors = self._build_author_errors(df_score, self.df_rows)

            # =========================
            # VOTE (same as score here but semantically separate)
            # =========================

            df_vote = (
                df_pairs.sort_values("score", ascending=False)
                .groupby("author_i")
                .head(1)
                .reset_index(drop=True)
            )

            vote_acc = (df_vote["author_i"] == df_vote["author_j"]).mean()

            df_vote_errors = self._build_author_errors(df_vote, self.df_rows)

            # =========================
            # BUILD RESULT
            # =========================
            strategy_results[strategy] = PairStrategyMetrics(
                patch=EvaluatedPrediction(
                    prediction=PredictionResult(data=df_patch, accuracy=patch_acc),
                    errors=ErrorAnalysis(data=df_patch_errors),
                ),
                votes=EvaluatedPrediction(
                    prediction=PredictionResult(data=df_vote, accuracy=vote_acc),
                    errors=ErrorAnalysis(data=df_vote_errors),
                ),
                score=EvaluatedPrediction(
                    prediction=PredictionResult(data=df_score, accuracy=score_acc),
                    errors=ErrorAnalysis(data=df_score_errors),
                ),
            )

        return PairEvaluationReport(
            df_rows=self.df_rows,
            df_pairs=self.df_pairs,
            strategy_results=strategy_results,
        )

    def _build_dataframe(self, threshold: float = 0.5) -> pd.DataFrame:
        if self.dataset is None or self.steps is None:
            raise ValueError("Need to initialize previous dataset and steps")

        rows = []

        for (img1, img2), (idx1, idx2, author1, author2, target) in self.dataset.take(
            self.steps
        ):
            probs = self.siamese_network([img1, img2], training=False)

            for i in range(len(probs)):

                rows.append(
                    {
                        "idx_i": int(idx1[i]),
                        "idx_j": int(idx2[i]),
                        "img1": img1[i],
                        "img2": img2[i],
                        "author_i": int(author1[i]),
                        "author_j": int(author2[i]),
                        "y_true": int(target[i]),
                        "prob": float(probs[i][0]),
                    }
                )

        df = pd.DataFrame(rows)
        df["y_pred"] = (df["prob"] >= threshold).astype(int)

        return df

    def _build_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        df_pairs = (
            df.groupby(["author_i", "author_j"])
            .agg(
                y_true=("y_true", "first"),
                prob_mean=("prob", "mean"),
                prob_sum=("prob", "sum"),
                prob_std=("prob", "std"),
                votes=("y_pred", "sum"),
                total=("prob", "size"),
            )
            .reset_index()
        )
        df_pairs["prob_std"] = df_pairs["prob_std"].fillna(0)
        df_pairs["vote_ratio"] = df_pairs["votes"] / df_pairs["total"]

        return df_pairs

    def _build_author_errors(
        self,
        df_best: pd.DataFrame,
        df_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Author-level errors → return all contributing rows
        """
        df_errors = df_best[df_best["author_i"] != df_best["author_j"]]

        error_pairs = set(zip(df_errors["author_i"], df_errors["author_j"]))

        return df_rows[
            df_rows.apply(
                lambda r: (r["author_i"], r["author_j"]) in error_pairs,
                axis=1,
            )
        ]


"""
class PairEvaluator:

    def __init__(
        self,
        siamese_network: tf.keras.Model,
    ) -> None:
        self.siamese_network = siamese_network

        self.dataset: tf.data.Dataset | None = None
        self.steps: int | None = None

        self.df_rows: pd.DataFrame | None = None
        self.df_pairs: pd.DataFrame | None = None

        self.score_functions: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
            "prob": lambda df: df["prob_mean"],
            "votes": lambda df: df["vote_ratio"],
            "hybrid": lambda df: df["vote_ratio"] * df["prob_mean"],
            "sum": lambda df: df["prob_sum"],
        }

    def update_data(
        self,
        dataset: tf.data.Dataset,
        steps: int,
        threshold: float = 0.5,
    ) -> None:
        self.dataset = dataset
        self.steps = steps

        self.df_rows = self._build_dataframe(threshold)
        self.df_pairs = self._build_pairs(self.df_rows)

    def evaluate(
        self, strategy: str = "prob"
    ) -> tuple[pd.DataFrame, pd.DataFrame, float]:
        self._validate_state()

        if strategy not in self.score_functions:
            raise ValueError(f"Invalid strategy: {strategy}")

        df_pairs = self.df_pairs.copy()
        df_pairs["score"] = self.score_functions[strategy](df_pairs)

        df_best = df_pairs.loc[
            df_pairs.groupby("author_i")["score"].idxmax()
        ].reset_index(drop=True)

        df_errors = self._build_errors(df_best)

        accuracy = (df_best["author_i"] == df_best["author_j"]).mean()

        return df_best, df_errors, accuracy

    def top_k_accuracy(self, k: int = 3, strategy: str = "sum") -> float:
        self._validate_state()

        if strategy not in self.score_functions:
            raise ValueError(f"Invalid strategy: {strategy}")

        df = self.df_pairs.copy()
        df["score"] = self.score_functions[strategy](df)
        correct = 0

        for _, group in df.groupby("author_i"):
            top_k = group.sort_values("score", ascending=False).head(k)

            if any(top_k["author_i"] == top_k["author_j"]):
                correct += 1

        return correct / df["author_i"].nunique()

    def _build_dataframe(self, threshold: float = 0.5) -> pd.DataFrame:
        if self.dataset is None or self.steps is None:
            raise ValueError("Need to initialize previous dataset and steps")

        rows = []

        for (img1, img2), (idx1, idx2, author1, author2, target) in self.dataset.take(
            self.steps
        ):
            probs = self.siamese_network([img1, img2], training=False)

            idx1 = idx1.numpy()
            idx2 = idx2.numpy()
            author1 = author1.numpy()
            author2 = author2.numpy()
            target = target.numpy()
            probs = probs.numpy()

            for i in range(len(probs)):
                prob = float(probs[i][0])

                rows.append(
                    {
                        "idx_i": int(idx1[i]),
                        "idx_j": int(idx2[i]),
                        "img1": img1[i],
                        "img2": img2[i],
                        "author_i": int(author1[i]),
                        "author_j": int(author2[i]),
                        "y_true": int(target[i]),
                        "prob": prob,
                    }
                )

        df_rows = pd.DataFrame(rows)

        df_rows["y_pred"] = (df_rows["prob"] >= threshold).astype(int)

        return df_rows

    def _build_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        df_pairs = (
            df.groupby(["author_i", "author_j"])
            .agg(
                y_true=("y_true", "first"),
                prob_mean=("prob", "mean"),
                prob_sum=("prob", "sum"),
                prob_std=("prob", "std"),
                votes=("y_pred", "sum"),
                total=("prob", "size"),
            )
            .reset_index()
        )
        df_pairs["prob_std"] = df_pairs["prob_std"].fillna(0)
        df_pairs["vote_ratio"] = df_pairs["votes"] / df_pairs["total"]

        return df_pairs

    def _build_errors(self, df_best: pd.DataFrame) -> pd.DataFrame:
        df_errors = df_best[df_best["author_i"] != df_best["author_j"]]

        error_pairs = set(zip(df_errors["author_i"], df_errors["author_j"]))

        return self.df_rows[
            self.df_rows.apply(
                lambda r: (r["author_i"], r["author_j"]) in error_pairs, axis=1
            )
        ]

    def _validate_state(self) -> None:
        if self.df_rows is None or self.df_pairs is None:
            raise ValueError(
                "Data not initialized. Call update_data() before evaluation."
            )
"""
