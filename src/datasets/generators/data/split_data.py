from dataclasses import dataclass

import numpy as np


@dataclass
class SplitData:
    X: np.ndarray  # (N, H, W) uint8
    y: np.ndarray  # (N,) int32
    author_index: dict[int, tuple[int, int]]  # {author_id: (start, end)}

    @property
    def authors_ids(self) -> list[int]:
        return list(self.author_index.keys())

    @property
    def valid_authors(self) -> list[int]:
        return [
            author_id
            for author_id, (start, end) in self.author_index.items()
            if end - start >= 2
        ]

    @property
    def num_patches(self) -> int:
        return len(self.X)
