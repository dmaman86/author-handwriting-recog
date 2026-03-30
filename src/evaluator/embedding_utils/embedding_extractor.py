import numpy as np
import pandas as pd
import tensorflow as tf

from ...generators import DataGenerator
from ...io import LoggerFactory


class EmbeddingExtractor:

    def __init__(
        self,
        model: tf.keras.Model,
        logger: LoggerFactory,
    ):
        self.model = model
        self.logger = logger

    def extract(
        self,
        data_gen: DataGenerator,
    ) -> pd.DataFrame:
        self.logger.info("Extracting embeddings from the model")

        dataset = data_gen.build()
        steps = data_gen.steps_per_epoch

        self.logger.info(f"Processing {steps} batches to extract embeddings")

        embeddings = []
        labels = []
        indices = []

        for batch, metadata in dataset.take(steps):
            emb = self.model(batch, training=False)
            idx, label = metadata
            embeddings.append(emb.numpy())
            labels.append(label.numpy())
            indices.append(idx.numpy())

        X = np.concatenate(embeddings)
        y = np.concatenate(labels)
        idxs = np.concatenate(indices)

        self.logger.info(f"Extracted embeddings for {len(X)} samples")

        df = pd.DataFrame(
            {
                "idx": idxs,
                "author": y,
                "embedding": list(X),
            }
        )
        return df
