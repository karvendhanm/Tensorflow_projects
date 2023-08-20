import tensorflow as tf

from tensorflow import keras


# subclassing a layer
class SimpleDense(keras.layers.Layer):

    def __init__(self, units, activation):
        """

        :param units: number of outputs from a layer.
        :param activation:
        """
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        input_dim = input_shape[-1]



