from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ..datasets import DatasetSplit
from ..io.logging import LoggerFactory
from .evaluator import EmbeddingAnalysisReport


class EmbeddingVisualizer:

    def __init__(
        self,
        report: EmbeddingAnalysisReport,
        logger: LoggerFactory,
        split: DatasetSplit,
    ) -> None:
        self._strategy: str | None = None
        self._results = None
        self.logger = logger
        self.split = split

        self.df_base = report.df_embeddings
        self.strategy_results = report.strategy_results

    def set_strategy(self, strategy: str):
        if strategy not in self.strategy_results:
            raise ValueError(f"Strategy '{strategy}' not found in report")

        self._strategy = strategy
        self._results = {
            level: result.data
            for level, result in self.strategy_results[strategy].items()
        }

    @property
    def strategy(self):
        return self._strategy

    @property
    def results(self):
        return self._results

    # -------------------------
    # Internal helpers
    # -------------------------
    def _author_name(self, author_id: int) -> str:
        return self.split.author_names[
            np.where(self.split.author_ids == author_id)[0][0]
        ]

    def _get_df(self, level: str) -> pd.DataFrame:
        if level not in self.results:
            raise ValueError(f"Unknown level: {level}")
        return self.results[level]

    def _require_predictions(self, df: pd.DataFrame, level: str):
        if "author_pred" not in df.columns:
            raise ValueError(f"{level} does not contain predictions")

    # -------------------------
    # Embedding space
    # -------------------------
    def plot_embeddings_2d(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(
            self.df_base["x"],
            self.df_base["y"],
            c=self.df_base["author"],
            s=10,
        )
        plt.title(f"Embedding (UMAP) - {self.strategy}")
        plt.show()

    # -------------------------
    # Patch-only (images)
    # -------------------------
    def log_samples(self, n: int = 50):
        df = self._get_df("patch")

        if len(df) == 0:
            self.logger.info("Empty dataframe")
            return

        sample = df.sample(min(n, len(df)))

        self.logger.info(f"Showing {len(sample)} samples:")

        for row in sample.itertuples():
            true_name = self._author_name(row.author)

            if hasattr(row, "author_pred"):
                pred_name = self._author_name(row.author_pred)
                self.logger.info(f"[idx={row.idx}] {true_name} → {pred_name}")
            else:
                self.logger.info(f"[idx={row.idx}] {true_name}")

    def log_error_samples(self, n: int = 50):
        df = self._get_df("patch")
        df_errors = df[df["author"] != df["author_pred"]]

        if len(df_errors) == 0:
            self.logger.info("No errors found 🎉")
            return

        sample = df_errors.sample(min(n, len(df_errors)))

        self.logger.info(f"Showing {len(sample)} error samples:")

        for row in sample.itertuples():
            true_name = self._author_name(row.author)
            pred_name = self._author_name(row.author_pred)

            self.logger.info(
                f"[idx={row.idx}] true={true_name} pred={pred_name} "
                f"(author_id={row.author} → {row.author_pred})"
            )

    # -------------------------
    # Confusion matrix
    # -------------------------
    def plot_confusion(self, level: str = "vote", remove_diagonal: bool = False):

        df = self._get_df(level)
        self._require_predictions(df, level)

        authors = sorted(df["author"].unique())

        cm = confusion_matrix(
            df["author"],
            df["author_pred"],
            labels=authors,
        )

        if remove_diagonal:
            np.fill_diagonal(cm, 0)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, cmap="Blues", linewidths=0.5)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix ({level}) - {self.strategy}")
        plt.show()

    def show_nearest(self, df_idx: int, k: int = 5):
        row = self.df_base.iloc[df_idx]
        query_emb = np.array(row.embedding)

        X = np.stack(self.df_base["embedding"].values)
        sims = X @ query_emb
        top_k = np.argsort(-sims)[:k]

        query_author_id = row.author
        query_author = self._author_name(query_author_id)

        self.logger.info(f"Query idx={row.idx} | author={query_author}")
        self.logger.info("Top-K nearest:")

        for rank, idx_df in enumerate(top_k):
            row_k = self.df_base.iloc[idx_df]

            author_id = row_k.author
            author_name = self._author_name(author_id)
            sim = sims[idx_df]

            match = "✓" if author_id == query_author_id else "✗"

            self.logger.info(
                f"{rank+1}. [idx={row_k.idx}] {author_name} "
                f"| sim={sim:.4f} | {match}"
            )

    # -------------------------
    # Centroids
    # -------------------------
    def plot_centroids(self, highlight: Iterable[int] | None = None):

        # Compute centroids on demand
        centroids = self.df_base.groupby("author")[["x", "y"]].mean().reset_index()

        plt.figure(figsize=(8, 6))

        # Base scatter
        plt.scatter(
            self.df_base["x"],
            self.df_base["y"],
            c=self.df_base["author"],
            s=10,
            alpha=0.5,
        )

        # Centroids
        plt.scatter(
            centroids["x"],
            centroids["y"],
            c="black",
            s=120,
            label="centroids",
        )

        # Labels
        for row in centroids.itertuples():
            plt.text(row.x, row.y, str(row.author))

        # Highlight (optional)
        if highlight is not None:
            highlight_set = set(highlight)

            sub = centroids[centroids["author"].isin(highlight_set)]

            plt.scatter(
                sub["x"],
                sub["y"],
                c="red",
                s=180,
                label="highlight",
            )

        plt.title(f"Centroids - {self.strategy}")
        plt.legend()
        plt.show()
