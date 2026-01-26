from typing import Union
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(
        self,
        data_partion: Union[dict[int, list[np.ndarray]], tuple[np.ndarray, np.ndarray]],
        batch_size: int = 32,
        target_size: tuple[int, int] = (60, 53),
        shuffle: bool = True,
        use_augmentation: bool = True,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.batch_size: int = batch_size
        self.target_size = target_size
        self.shuffle: bool = shuffle
        self.use_augmentation: bool = use_augmentation
        self.seed: int | None = seed

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

        if isinstance(data_partion, dict):
            self._init_from_dict(data_partion)
        elif isinstance(data_partion, tuple) and len(data_partion) == 2:
            self._init_from_arrays(data_partion)
        else:
            raise TypeError(
                f"data_partion must be dict or tuple, got {type(data_partion)}"
            )

        self._prepare_data()

        self._augmentation = self._build_augmentation() if use_augmentation else None
        
        self._create_internal_dataset()

        self.on_epoch_end()

    def _prepare_data(self) -> None:
        if self.X.max() > 1.0:
            self.X = self.X / 255.0

        if len(self.X.shape) == 3:
            self.X = np.expand_dims(self.X, axis=-1)

    def _create_internal_dataset(self) -> None:
        dataset = tf.data.Dataset.from_tensor_slices((self.X, self.y))
        
        if self.shuffle:
            dataset = dataset.shuffle(
                buffer_size=len(self.X),
                seed=self.seed,
                reshuffle_each_iteration=True
                )

        if self.use_augmentation and self._augmentation is not None:
            def augment_fn(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
                images = self._augmentation(images, training=True)
                return images, labels

            dataset = dataset.map(
                augment_fn,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        def resise_fn(img: tf.Tensor, lbl: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            img = tf.image.resize(img, self.target_size)
            return img, lbl

        dataset = dataset.map(resise_fn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)

        self._dataset = dataset
        self._iterator = iter(dataset)

    def _init_from_dict(self, data_partition: dict[int, list[np.ndarray]]) -> None:
        X: list[np.ndarray] = []
        y: list[int] = []

        for author_id, patches in data_partition.items():
            for patch in patches:
                X.append(patch)
                y.append(author_id)

        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int32)

    def _init_from_arrays(
        self, data_partition: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = data_partition
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int32)

    def _build_augmentation(self) -> tf.keras.Sequential:
        """
        Defines augmentation suitable for handwriting patches.
        """
        return tf.keras.Sequential(
            [
                layers.RandomRotation(0.02),
                layers.RandomZoom(0.1),
                layers.RandomTranslation(0.05, 0.05),
                layers.RandomContrast(0.1),
            ],
            name="classification_augmentation",
        )

    def __len__(self) -> int:
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        try:
            batch_x, batch_y = next(self._iterator)
            return batch_x.numpy(), batch_y.numpy()
        except StopIteration:
            self._iterator = iter(self._dataset)
            batch_x, batch_y = next(self._iterator)
            return batch_x.numpy(), batch_y.numpy()
        
    def on_epoch_end(self) -> None:
        self._iterator = iter(self._dataset)
    