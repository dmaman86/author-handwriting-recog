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

        preds = df.groupby("author")["author_pred"].agg(
            lambda x: x.value_counts().idxmax()
        )
        result = preds.reset_index()
        result.columns = ["author", "author_pred"]

        result["correct"] = result["author"] == result["author_pred"]

        return result


class ScoreAggregator(Aggregator):

    name = "score"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped = df.groupby("author")

        preds = grouped.apply(
            lambda g: g.groupby("author_pred")["score"].mean().idxmax()
        )

        result = preds.reset_index()
        result.columns = ["author", "author_pred"]

        result["correct"] = result["author"] == result["author_pred"]

        return result
