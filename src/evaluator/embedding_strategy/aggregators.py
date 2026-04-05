from abc import ABC, abstractmethod

import pandas as pd


class Aggregator(ABC):

    name: str

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class PatchAggregator(Aggregator):

    name = "patch"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["correct"] = df["author"] == df["author_pred"]
        return df


class VoteAggregator(Aggregator):

    name = "vote"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:

        counts = df.groupby(["author", "author_pred"]).size().reset_index(name="count")

        best_idx = counts.groupby("author")["count"].idxmax()
        result = counts.loc[best_idx, ["author", "author_pred"]].reset_index(drop=True)

        result["correct"] = result["author"] == result["author_pred"]

        return result


class ScoreAggregator(Aggregator):

    name = "score"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:

        summed_scores = (
            df.groupby(["author", "author_pred"])["score"].sum().reset_index()
        )

        best_preds_idx = summed_scores.groupby("author")["score"].idxmax()

        result = summed_scores.loc[
            best_preds_idx, ["author", "author_pred"]
        ].reset_index(drop=True)

        result["correct"] = result["author"] == result["author_pred"]

        return result
