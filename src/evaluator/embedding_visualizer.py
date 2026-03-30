from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ..datasets import ZarrLoader
from ..io.logging import LoggerFactory
from .evaluator import EmbeddingAnalysisReport


class EmbeddingVisualizer:

    def __init__(
        self,
        report: EmbeddingAnalysisReport,
        logger: LoggerFactory,
        loader: ZarrLoader,
    ) -> None:
        self._strategy: str | None = None
        self._results = None
        self.logger = logger
        self.loader = loader

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
    def _get_image(self, idx: int) -> np.ndarray:
        img = self.loader.images[idx]
        return img.squeeze()

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
    # Errors
    # -------------------------
    def plot_errors_2d(self, level: str = "patch"):
        df = self._get_df(level).copy()
        self._require_predictions(df, level)

        df["is_error"] = df["author"] != df["author_pred"]
        colors = df["is_error"].map({True: "red", False: "blue"})

        plt.figure(figsize=(8, 6))
        plt.scatter(self.df_base["x"], self.df_base["y"], c=colors, s=10)
        plt.title(f"Errors ({level}) - {self.strategy}")
        plt.show()

    def plot_author_errors_2d(self):
        df_vote = self._get_df("vote")
        bad_authors = set(df_vote[df_vote["correct"] == False]["author"])

        df = self.df_base.copy()
        df["author_error"] = df["author"].isin(bad_authors)

        colors = df["author_error"].map({True: "red", False: "blue"})

        plt.figure(figsize=(8, 6))
        plt.scatter(df["x"], df["y"], c=colors, s=10)
        plt.title(f"Author-level Errors - {self.strategy}")
        plt.show()

    # -------------------------
    # Patch-only (images)
    # -------------------------
    def plot_with_images(self, n=50):
        df = self._get_df("patch")
        sample = df.sample(n)

        plt.figure(figsize=(10, 10))
        for i, row in enumerate(sample.itertuples()):
            idx = int(row.idx)
            img = self._get_image(idx)

            plt.subplot(10, 5, i + 1)
            plt.imshow(img, cmap="gray")
            plt.title(f"{row.author}->{row.author_pred}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def plot_error_images(self, n=50):
        df = self._get_df("patch")
        df_errors = df[df["author"] != df["author_pred"]]

        if len(df_errors) == 0:
            self.logger.info("No errors found 🎉")
            return

        sample = df_errors.sample(min(n, len(df_errors)))

        plt.figure(figsize=(10, 10))
        for i, row in enumerate(sample.itertuples()):
            idx = int(row.idx)
            img = self._get_image(idx)

            plt.subplot(10, 5, i + 1)
            plt.imshow(img, cmap="gray")
            plt.title(f"{row.author} → {row.author_pred}", color="red")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    # -------------------------
    # Author-level analysis
    # -------------------------
    def plot_author_accuracy(self):
        df = self._get_df("vote").copy()
        df["correct"] = df["correct"].astype(int)

        plt.figure(figsize=(10, 4))
        plt.bar(df["author"], df["correct"])
        plt.title(f"Author Accuracy (vote) - {self.strategy}")
        plt.xlabel("Author")
        plt.ylabel("Correct")
        plt.show()

    def plot_vote_vs_score_disagreement(self):
        vote_df = self._get_df("vote")
        score_df = self._get_df("score")

        df = vote_df.merge(
            score_df,
            on="author",
            suffixes=("_vote", "_score"),
        )

        disagreements = df[df["author_pred_vote"] != df["author_pred_score"]]

        if len(disagreements) == 0:
            self.logger.info("No disagreement between vote and score 🎉")
            return

        self.logger.info("Disagreements:")
        self.logger.info(f"\n{disagreements.head()}")

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

        plt.figure(figsize=(k * 3, 3))

        for i, idx_df in enumerate(top_k):
            idx = int(self.df_base.iloc[idx_df]["idx"])
            img = self._get_image(idx)

            plt.subplot(1, k, i + 1)
            plt.imshow(img, cmap="gray")
            plt.title(f"sim={sims[idx_df]:.2f}")
            plt.axis("off")

        plt.show()

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
