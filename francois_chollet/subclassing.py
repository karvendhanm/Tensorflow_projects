import tensorflow as tf

from tensorflow import keras


# subclassing a layer
class DenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None):
        """

        :param units:
        :param activation:
        """
        super().__init__()
        self.W = None
        self.b = None
        self.units = units
        self.activation = activation


    def build(self, input_shape):
        """

        :param input_dim:
        :return:
        """
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units), initializer='random_normal')
        self.b = self.add_weight(shape=(self.units, ), initializer='zeros')


    def call(self, inputs):
        """

        :param inputs:
        :return:
        """
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y


denselayer = DenseLayer(32, tf.nn.relu)
input_tensor = tf.ones(shape=(2, 784))
output_tensor = denselayer(input_tensor)
print(output_tensor.shape)




