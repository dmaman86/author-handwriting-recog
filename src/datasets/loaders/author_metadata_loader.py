from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AuthorInfo:
    author_id: str
    lines: list[tuple[int, int]]
    test_area_top: int
    test_area_bottom: int
    scale_factor: float
    line_height: int

    @property
    def num_lines(self) -> int:
        return len(self.lines)

    @property
    def first_line_y(self) -> int:
        return self.lines[0][0]

    @property
    def last_line_y(self) -> int:
        return self.lines[-1][1]


class AuthorMetadataLoader:
    def __init__(
        self,
        parent_path: str | Path,
        load_fn: Callable[[Path], dict],
    ) -> None:
        self.parent_path = Path(parent_path)
        self.load_fn = load_fn

    def build_author_data(self, author_id: str) -> AuthorInfo:
        mat_path = self.parent_path / f"{author_id}.mat"

        if not mat_path.exists():
            raise FileNotFoundError(f"Missing .mat file for author {author_id}")

        mat_info = self.load_fn(mat_path)

        peaks: np.ndarray = mat_info["peaks_indices"].flatten()
        scale: float = float(mat_info["SCALE_FACTOR"].flatten()[0])
        top: int = int(mat_info["top_test_area"].flatten()[0])
        bottom: int = int(mat_info["bottom_test_area"].flatten()[0])

        peaks_scaled = [int(p * scale) for p in peaks]

        if len(peaks_scaled) < 2:
            raise ValueError(f"Invalid peaks for author {author_id}")

        lines = [
            (peaks_scaled[i], peaks_scaled[i + 1]) for i in range(len(peaks_scaled) - 1)
        ]

        return AuthorInfo(
            author_id=author_id,
            lines=lines,
            test_area_top=top,
            test_area_bottom=bottom,
            scale_factor=scale,
            line_height=bottom - top,
        )
