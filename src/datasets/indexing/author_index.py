from collections import defaultdict

import numpy as np


def build_author_index(labels: np.ndarray) -> dict[int, list[int]]:
    index = defaultdict(list)

    for i, a in enumerate(labels):
        index[int(a)].append(i)
    return dict(index)
