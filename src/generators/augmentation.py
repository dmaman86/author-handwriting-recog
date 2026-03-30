from typing import Callable

import tensorflow as tf


class AugmentationOps:

    def __init__(
        self,
        rotation_range: float = 0.0,
        width_shift_range: float = 0.0,
        height_shift_range: float = 0.0,
        horizontal_flip: bool = False,
        resize_dim: tuple[int, int] | None = None,
        brightness_delta: float = 0.05,
        contrast_range: tuple[float, float] = (0.95, 1.05),
        noise_stddev: float = 0.015,
        zoom_range: tuple[float, float] = (0.9, 1.1),
    ) -> None:
        # fraction_rot * 360 = 0.05 * 360 = 18 degrees
        self.rot_layer = tf.keras.layers.RandomRotation(rotation_range / 360)

        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.horizontal_flip = horizontal_flip
        self.resize_dim = resize_dim
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.noise_stddev = noise_stddev
        self.zoom_range = zoom_range

    def rotate(self, image: tf.Tensor) -> tf.Tensor:
        return self.rot_layer(image)

    def brightness_contrast(self, image: tf.Tensor) -> tf.Tensor:
        image = tf.image.random_brightness(image, max_delta=self.brightness_delta)
        image = tf.image.random_contrast(
            image,
            lower=self.contrast_range[0],
            upper=self.contrast_range[1],
        )
        return image

    def gaussian_noise(self, image: tf.Tensor) -> tf.Tensor:
        noise = tf.random.normal(
            tf.shape(image),
            mean=0.0,
            stddev=self.noise_stddev * 255.0,
        )
        return image + noise

    def zoom(self, image: tf.Tensor) -> tf.Tensor:
        # image: (B, H, W, C)
        shape = tf.shape(image)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]

        scale = tf.random.uniform([B], self.zoom_range[0], self.zoom_range[1])

        new_h = tf.cast(tf.cast(H, tf.float32) * scale, tf.int32)
        new_w = tf.cast(tf.cast(W, tf.float32) * scale, tf.int32)

        def resize_one(args):
            img, nh, nw = args
            img = tf.image.resize(img, (nh, nw))
            img = tf.image.resize_with_crop_or_pad(img, H, W)
            return img

        return tf.map_fn(
            resize_one,
            (image, new_h, new_w),
            fn_output_signature=tf.float32,
        )

    def shift(self, image: tf.Tensor) -> tf.Tensor:
        # image: (B, H, W, C)
        shape = tf.shape(image)
        B, H, W, C = shape[0], shape[1], shape[2], shape[3]

        max_dx = tf.cast(self.height_shift_range * tf.cast(H, tf.float32), tf.int32)
        max_dy = tf.cast(self.width_shift_range * tf.cast(W, tf.float32), tf.int32)

        dx = tf.random.uniform([B], -max_dx, max_dx + 1, dtype=tf.int32)
        dy = tf.random.uniform([B], -max_dy, max_dy + 1, dtype=tf.int32)

        image = tf.pad(
            image,
            [[0, 0], [max_dx, max_dx], [max_dy, max_dy], [0, 0]],
        )

        def crop_one(args):
            img, dx_i, dy_i = args
            return tf.image.crop_to_bounding_box(
                img,
                max_dx + dx_i,
                max_dy + dy_i,
                H,
                W,
            )

        return tf.map_fn(
            crop_one,
            (image, dx, dy),
            fn_output_signature=tf.float32,
        )

    def flip(self, image: tf.Tensor) -> tf.Tensor:
        if self.horizontal_flip:
            image = tf.image.random_flip_left_right(image)
        return image

    def resize(self, image: tf.Tensor) -> tf.Tensor:
        if self.resize_dim is None:
            return image

        return tf.image.resize(image, self.resize_dim)

    def to_rgb(self, image: tf.Tensor) -> tf.Tensor:
        return tf.image.grayscale_to_rgb(image)

    def clip_255(self, image: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(image, 0.0, 255.0)

    def efficientnet_preprocess(self, image: tf.Tensor) -> tf.Tensor:
        return tf.keras.applications.efficientnet_v2.preprocess_input(image)

    def mobilenet_preprocess(self, image: tf.Tensor) -> tf.Tensor:
        return tf.keras.applications.mobilenet_v2.preprocess_input(image)

    def custom_preprocess(self, image: tf.Tensor) -> tf.Tensor:
        return image / 255.0


class AugmentationPipeline:

    def __init__(self, ops: list[Callable[[tf.Tensor], tf.Tensor]]) -> None:
        self.ops = ops

    def apply(self, image: tf.Tensor) -> tf.Tensor:
        for op in self.ops:
            image = op(image)
        return image


class AugmentImage:

    def __init__(self, pipeline: AugmentationPipeline) -> None:
        self.pipeline = pipeline

    def __call__(self, inputs, metadata):
        outputs = tf.nest.map_structure(
            lambda x: self.pipeline.apply(x),
            inputs,
        )
        return outputs, metadata
