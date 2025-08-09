import tensorflow as tf

def build_model(input_shape: tuple[int, int, int], num_classes: int) -> tf.keras.Model:
  model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),

    tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

    tf.keras.layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

    tf.keras.layers.Conv2D(256, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),

    tf.keras.layers.Conv2D(512, kernel_size=3, padding="same", activation="relu"),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dropout(0.6),

    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")
  ])
  return model

def get_activation_model(model: tf.keras.Model) -> tuple[tf.keras.Model, list[str]]:
  """
    Build a model that outputs activations from all Conv2D layers.

    Args:
        model (tf.keras.Model): Trained Keras model.

    Returns:
        tuple: (intermediate model, list of layer names)
  """
  conv_layers = [layer for layer in model.layers if layer.name.startswith("conv2d")]
  outputs = [layer.output for layer in conv_layers]
  layer_names = [layer.name for layer in conv_layers]
  activation_model = tf.keras.Model(inputs=model.layers[0].input, outputs=outputs)
  return activation_model, layer_names


def text_callable(layer_index:int, layer: tf.keras.layers.Layer) -> tuple[str, bool]:
  above = bool(layer_index % 2)

  try:
    output_shape = [x for x in list(layer.output.shape) if x is not None]
  except AttributeError:
    output_shape = layer.output.shape.as_list()

  if isinstance(output_shape[0], tuple):
    output_shape = list(output_shape[0])
    output_shape = [x for x in output_shape if x is not None]

  txt_parts = []
  for i, dim in enumerate(output_shape):
    txt_parts.append(str(dim))
    if i < len(output_shape) - 2:
      txt_parts.append('x')
    if i == len(output_shape) - 2:
      txt_parts.append('\n')
  shape_txt = ''.join(txt_parts)
  shape_txt += f'\n{layer.name}'
  return shape_txt, above