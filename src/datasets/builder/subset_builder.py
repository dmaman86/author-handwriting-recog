import gc
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..generators.data.split_data import SplitData
from ..loaders.selective_data_loader import SelectiveDataLoader

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class SubsetBuildConfig:
    splits: tuple[SplitName, ...] = ("train", "validation", "test")
    gc_collect: bool = True


@dataclass
class SubsetResult:
    splits: dict[SplitName, SplitData]
    num_authors: int
    author_names: list[str]
    patch_shape: tuple[int, int, int]
    global_to_local: dict[int, int]
    local_to_global: dict[int, int]


class SubsetBuilder:
    def __init__(
        self,
        loader: SelectiveDataLoader,
        config: SubsetBuildConfig | None = None,
    ) -> None:
        self._loader = loader
        self._config = config or SubsetBuildConfig()

    def build(
        self, authors_range: list[int] | range, show_progress_bar: bool = True
    ) -> SubsetResult:
        if isinstance(authors_range, range):
            authors_range = list(authors_range)

        raw_subset = self._loader.load_dataset_by_authors(
            authors_id=authors_range,
            show_progress_bar=show_progress_bar,
        )

        splits: dict[SplitName, SplitData] = {}

        for split in self._config.splits:
            X, y = raw_subset.pop(split)
            author_index = self._build_author_index(y)
            splits[split] = SplitData(X=X, y=y, author_index=author_index)
            gc.collect()

        result = SubsetResult(
            splits=splits,
            num_authors=raw_subset["num_authors"],
            author_names=raw_subset["author_names"],
            patch_shape=raw_subset["patch_shape"],
            global_to_local=raw_subset["global_to_local"],
            local_to_global=raw_subset["local_to_global"],
        )

        del raw_subset
        if self._config.gc_collect:
            gc.collect()

        return result

    @staticmethod
    def _build_author_index(y: np.ndarray) -> dict[int, tuple[int, int]]:
        author_index: dict[int, tuple[int, int]] = {}
        current_author = None
        start = 0

        for i, aid in enumerate(y):
            aid = int(aid)
            if aid != current_author:
                if current_author is not None:
                    author_index[current_author] = (start, i)
                current_author = aid
                start = i

        if current_author is not None:
            author_index[current_author] = (start, len(y))

        return author_index
