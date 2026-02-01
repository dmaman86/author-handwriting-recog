from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..generators.data.author_partition_state import AuthorPartitionState
from ..generators.data.data_initializer import DataInitializer
from ..generators.data.data_normalizer import PatchNormalizer
from ..loaders.selective_data_loader import SelectiveDataLoader

SplitName = Literal["train", "validation", "test"]
RawSplit = tuple[np.ndarray, np.ndarray]  # (X, y)


@dataclass(frozen=True)
class SubsetBuildConfig:
    splits: tuple[SplitName, ...] = ("train", "validation", "test")
    gc_collect: bool = True


class SubsetBuilder:
    def __init__(
        self,
        loader: SelectiveDataLoader,
        data_normalizer: PatchNormalizer | None = None,
        config: SubsetBuildConfig | None = None,
    ) -> None:
        self._loader = loader
        self._data_normalizer = data_normalizer
        self._config = config or SubsetBuildConfig()

    def build(self, authors_range: list[int] | range, show_progress_bar: bool = True):
        raw_subset = self._loader.load_dataset_by_authors(
            authors_id=authors_range, show_progress_bar=show_progress_bar
        )

        result: dict[SplitName, AuthorPartitionState] = {}

        for split in self._config.splits:
            raw_split: RawSplit = raw_subset[split]

            initializer = DataInitializer(raw_split)
            state = initializer.to_state()
            if self._data_normalizer:
                state.data = self._data_normalizer.normalize_partition(data=state.data)

            result[split] = state
            del raw_split
            gc.collect()

        del raw_subset
        if self._config.gc_collect:
            gc.collect()
        return result
