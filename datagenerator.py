import tensorflow as tf
import numpy as np
import os

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(
      self,
      list_ids: list[str],
      labels: dict[str, int],
      binary_dir: str,
      n_classes: int,
      dim: tuple[int, int, int] = (60, 53, 1),
      batch_size: int = 32,
      shuffle: bool = True,
      augment: bool = False,
      use_partial_dataset: bool = False,
      partial_ratio: float = 1.0,
      **kwargs
      ) -> None:
    super().__init__(**kwargs)

    self.list_ids = list_ids
    self.labels = labels
    self.binary_dir = binary_dir
    self.n_classes = n_classes
    self.batch_size = batch_size
    self.dim = dim
    self.shuffle = shuffle
    self.augment = augment
    self.use_partial_dataset = use_partial_dataset
    self.partial_ratio = partial_ratio

    if self.augment:
      self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
          rotation_range=10,
          width_shift_range=0.1,
          height_shift_range=0.1,
          fill_mode='nearest'
      )

    self.on_epoch_end()

  def __len__(self):
    return max(1, int(np.floor(len(self.list_ids) * self.partial_ratio) // self.batch_size))

  def __getitem__(self, index):
    # indexes of the batch
    batch_ids = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
    # list of filenames
    batch_files = [self.list_ids[k] for k in batch_ids]
    # generate data
    X, y = self.__data_generation(batch_files)
    return X, y

  def on_epoch_end(self):
    total_len = len(self.list_ids)
    if self.use_partial_dataset:
      partial_len = max(1, int(total_len * self.partial_ratio))
      self.indexes = np.random.choice(total_len, partial_len, replace=False)
    else:
      self.indexes = np.arange(total_len)

    if self.shuffle:
      np.random.shuffle(self.indexes)

  def _apply_augmentation(self, patch: np.ndarray) -> np.ndarray:
    if self.augment:
      patch = self.datagen.random_transform(patch)
    return patch

  def __data_generation(self, batch_files: list[str]) -> tuple[np.ndarray, np.ndarray]:
    # Initialization
    X = np.empty((self.batch_size, *self.dim), dtype=np.float32)
    y = np.empty((self.batch_size), dtype=int)
    target_size = self.dim[:2]  # (height, width)

    for i, filename in enumerate(batch_files):
      path = os.path.join(self.binary_dir, filename)
      patch = np.load(path).astype(np.float32) / 255.0  # normalize

      # Expand dims only if patch is missing channel dimension
      if patch.ndim == 2 and self.dim[-1] == 1:
        patch = np.expand_dims(patch, axis=-1)

      # resize to target size (height, width)
      patch = tf.image.resize(patch, size=target_size, method='bilinear').numpy()

      patch = self._apply_augmentation(patch)

      X[i] = patch
      y[i] = self.labels[filename]

    y = tf.keras.utils.to_categorical(y, num_classes=self.n_classes)
    return X, y