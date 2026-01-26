from typing import Union

import numpy as np


class DataInitializer:

    def __init__(
        self,
        data_partition: Union[
            dict[int, list[np.ndarray]], tuple[np.ndarray, np.ndarray]
        ],
    ) -> None:
        if isinstance(data_partition, dict):
            self._from_dict(data_partition)
        elif isinstance(data_partition, tuple) and len(data_partition) == 2:
            self._from_arrays(data_partition)
        else:
            raise TypeError(
                f"data_partition must be dict or tuple got {type(data_partition)}"
            )

    def _from_dict(self, data_partition: dict[int, list[np.ndarray]]) -> None:
        self.data = data_partition
        self.author_ids = list(data_partition.keys())
        self.valid_authors = [
            author_id for author_id, patches in self.data.items() if len(patches) >= 2
        ]
        self._validate()

    def _from_arrays(self, data_partition: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = data_partition

        self.data: dict[int, list[np.ndarray]] = {}

        for patch, author_id in zip(X, y):
            author_id = int(author_id)

            if author_id not in self.data:
                self.data[author_id] = []

            self.data[author_id].append(patch)

        self.authors_ids = list(self.data.keys())
        self.valid_authors: list[int] = [
            author_id for author_id, patches in self.data.items() if len(patches) >= 2
        ]

        self._validate()

    def _validate(self) -> None:
        if len(self.valid_authors) < 2:
            raise ValueError(
                "Data requires at least two authors " "with at least two patches each."
            )
