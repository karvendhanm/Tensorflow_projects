# Keras Dense layer implementation - creating state just in time

import tensorflow as tf
from tensorflow import keras

class SimpleDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units),
                                 initializer='random_normal')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros')

    def call(self, inputs):
        y = tf.matmul(inputs, self.W) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y


    def __call__(self, inputs):
        if not self.built:
            self.build(inputs.shape)
            self.built = True
        return self.call(inputs)

my_dense = SimpleDense(units=32, activation=tf.nn.relu)
input_tensor = tf.ones(shape=(2, 784))

# TODO understand how this call behaves unexpectedly must be understood.
# how does eager tensor (input_tensor) morph into tensor shape object (input_shape)
# Also, how this function call the build function and then call function should be understood
output_tensor = my_dense(input_tensor)
print(output_tensor.shape)

