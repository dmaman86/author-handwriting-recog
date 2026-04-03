from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

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

    def _get_umap_dims(self) -> int:
        return len([c for c in self.df_base.columns if c.startswith("umap_")])

    def _f(self, series: pd.Series) -> pd.Series:
        """Cast to float64 — plotly silently breaks with float32/int32."""
        return series.astype(float)

    # -------------------------
    # Embedding space
    # -------------------------
    def _plot_embeddings_2d(self):
        author_names = self.df_base["author"].map(self._author_name)
        fig = go.Figure(
            go.Scatter(
                x=self._f(self.df_base["umap_0"]),
                y=self._f(self.df_base["umap_1"]),
                mode="markers",
                marker=dict(
                    color=self._f(self.df_base["author"]),
                    colorscale="Viridis",
                    size=5,
                    showscale=True,
                ),
                text=author_names,
            )
        )
        fig.update_layout(
            title=f"Embedding (UMAP 2D) - {self.strategy}",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
        )
        fig.show()

    def _plot_embeddings_3d(self):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            self.df_base["umap_0"],
            self.df_base["umap_1"],
            self.df_base["umap_2"],
            c=self.df_base["author"],
            s=10,
            alpha=0.7,
        )

        ax.set_title(f"Embedding (UMAP 3D) - {self.strategy}")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        plt.show()

    def plot_embeddings(self):
        dims = self._get_umap_dims()

        if dims == 2:
            self._plot_embeddings_2d()
        elif dims == 3:
            self._plot_embeddings_3d()
        else:
            raise ValueError(f"Unsupported UMAP dimensions: {dims}")

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

    def _estimate_eps(self, X: np.ndarray, k: int = 2) -> float:

        if k < 2:
            k = max(2, min(5, int(np.log(len(X)))))

        nbrs = NearestNeighbors(n_neighbors=k).fit(X)

        distances, _ = nbrs.kneighbors(X)
        # distance to k-th nearest neighbor
        k_distances = np.sort(distances[:, -1])

        eps = np.percentile(k_distances, 90)

        return float(eps)

    # -------------------------
    # Centroids
    # -------------------------
    def _plot_centroids_2d(
        self, centroids: pd.DataFrame, highlight: Iterable[int] | None = None
    ):
        fig = go.Figure()

        if highlight is not None:
            highlight_set = set(highlight)

            # Only samples from highlighted authors
            df_sub = self.df_base[self.df_base["author"].isin(highlight_set)]
            fig.add_trace(
                go.Scatter(
                    x=self._f(df_sub["umap_0"]),
                    y=self._f(df_sub["umap_1"]),
                    mode="markers",
                    marker=dict(
                        color=self._f(df_sub["author"]),
                        colorscale="Viridis",
                        size=5,
                        opacity=0.5,
                    ),
                    hovertext=df_sub["author"].apply(
                        lambda a: f"{self._author_name(a)} (id={a})"
                    ),
                    hoverinfo="text",
                    name="samples",
                )
            )

            # Highlighted centroids
            sub = centroids[centroids["author"].isin(highlight_set)]
            fig.add_trace(
                go.Scatter(
                    x=self._f(sub["umap_0"]),
                    y=self._f(sub["umap_1"]),
                    mode="markers+text",
                    marker=dict(color="red", size=14, symbol="circle-open"),
                    text=sub["author"].apply(self._author_name),
                    textposition="top center",
                    hovertext=sub["author"].apply(
                        lambda a: f"{self._author_name(a)} (id={a})"
                    ),
                    hoverinfo="text",
                    name="highlight",
                )
            )
        else:
            # All samples
            fig.add_trace(
                go.Scatter(
                    x=self._f(self.df_base["umap_0"]),
                    y=self._f(self.df_base["umap_1"]),
                    mode="markers",
                    marker=dict(
                        color=self._f(self.df_base["author"]),
                        colorscale="Viridis",
                        size=5,
                        opacity=0.4,
                    ),
                    hovertext=self.df_base["author"].apply(
                        lambda a: f"{self._author_name(a)} (id={a})"
                    ),
                    hoverinfo="text",
                    name="samples",
                )
            )

            # Noise centroids (DBSCAN label == -1)
            noise = centroids[centroids["cluster"] == -1]
            if not noise.empty:
                fig.add_trace(
                    go.Scatter(
                        x=self._f(noise["umap_0"]),
                        y=self._f(noise["umap_1"]),
                        mode="markers",
                        marker=dict(color="gray", size=12, symbol="x"),
                        hovertext=noise["author"].apply(
                            lambda a: f"{self._author_name(a)} (id={a})"
                        ),
                        hoverinfo="text",
                        name="noise",
                    )
                )

            # Clustered centroids
            clustered = centroids[centroids["cluster"] >= 0]
            if not clustered.empty:
                fig.add_trace(
                    go.Scatter(
                        x=self._f(clustered["umap_0"]),
                        y=self._f(clustered["umap_1"]),
                        mode="markers",
                        marker=dict(
                            color=self._f(clustered["cluster"]),
                            colorscale="Rainbow",
                            size=12,
                            showscale=True,
                        ),
                        hovertext=clustered["author"].apply(
                            lambda a: f"{self._author_name(a)} (id={a})"
                        ),
                        hoverinfo="text",
                        name="centroids",
                    )
                )

        fig.update_layout(
            title=f"Centroids 2D + DBSCAN - {self.strategy}",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
        )
        fig.show()

    def _plot_centroids_3d(
        self, centroids: pd.DataFrame, highlight: Iterable[int] | None = None
    ):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        # Base scatter
        ax.scatter(
            self.df_base["umap_0"],
            self.df_base["umap_1"],
            self.df_base["umap_2"],
            c=self.df_base["author"],
            s=5,
            alpha=0.3,
        )

        if highlight is not None:
            sub = centroids[centroids["author"].isin(set(highlight))]
            ax.scatter(
                sub["umap_0"],
                sub["umap_1"],
                sub["umap_2"],
                c="red",
                s=120,
                label="highlight",
            )
            for row in sub.itertuples():
                ax.text(
                    row.umap_0,
                    row.umap_1,
                    row.umap_2,
                    self._author_name(row.author),
                    fontsize=8,
                )
        else:
            # Noise centroids (DBSCAN label == -1)
            noise = centroids[centroids["cluster"] == -1]
            if not noise.empty:
                ax.scatter(
                    noise["umap_0"],
                    noise["umap_1"],
                    noise["umap_2"],
                    c="gray",
                    s=80,
                    marker="x",
                    label="noise",
                )

            # Clustered centroids
            clustered = centroids[centroids["cluster"] >= 0]
            if not clustered.empty:
                ax.scatter(
                    clustered["umap_0"],
                    clustered["umap_1"],
                    clustered["umap_2"],
                    c=clustered["cluster"],
                    cmap="tab10",
                    s=80,
                    label="centroids",
                )

        ax.set_title(f"Centroids (3D) - {self.strategy}")
        ax.legend()
        plt.show()

    def plot_centroids(self, highlight: Iterable[int] | None = None):
        dims = self._get_umap_dims()

        cols = [f"umap_{i}" for i in range(dims)]

        centroids = self.df_base.groupby("author")[cols].mean().reset_index()
        X = centroids[cols].values

        eps = self._estimate_eps(X, k=2)

        db = DBSCAN(eps=eps, min_samples=2)
        labels = db.fit_predict(X)

        centroids["cluster"] = labels

        self.logger.info(
            f"Estimated DBSCAN eps={eps:.4f} | clusters={len(set(labels))}"
        )

        if dims == 2:
            self._plot_centroids_2d(centroids, highlight)
        elif dims == 3:
            self._plot_centroids_3d(centroids, highlight)
        else:
            raise ValueError(f"Unsupported UMAP dimensions: {dims}")
