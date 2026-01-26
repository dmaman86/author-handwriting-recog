import random


class PairIndexSampler:

    def __init__(
        self,
        author_index: dict[int, list[int]],
        positive_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.author_index = author_index
        self.author_ids = list(author_index.keys())
        self.valid_authors = [a for a, idxs in author_index.items() if len(idxs) >= 2]
        self.positive_ratio = positive_ratio

        if seed is not None:
            random.seed(seed)

    def sample(self, batch_size: int) -> tuple[list[tuple[int, int]], list[int]]:
        num_pos = int(batch_size * self.positive_ratio)
        num_neg = batch_size - num_pos

        pairs = []
        labels = []

        for _ in range(num_pos):
            a = random.choice(self.valid_authors)
            i1, i2 = random.sample(self.author_index[a], 2)
            pairs.append((i1, i2))
            labels.append(1)

        for _ in range(num_neg):
            a1, a2 = random.sample(self.author_ids, 2)
            i1 = random.choice(self.author_index[a1])
            i2 = random.choice(self.author_index[a2])
            pairs.append((i1, i2))
            labels.append(0)

        return pairs, labels
