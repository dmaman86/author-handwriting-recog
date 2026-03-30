from abc import ABC, abstractmethod
import tensorflow as tf

class BaseBackbone(ABC):

    @abstractmethod
    def build(self, inputs: tf.keras.Input) -> tf.Tensor:
        """
        Args:
            inputs: tf.keras.Input
        Returns:
            feature tensor (before embedding head)
        """
        pass