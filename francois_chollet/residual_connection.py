import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation='relu')(inputs)
residual = x
x = layers.Conv2D(64, 3, activation='relu', padding='same', name='second_convolutional_layer')(x)
residual = layers.Conv2D(64, 1, padding='same')(residual)
x = layers.add([x, residual])
outputs = layers.Conv2D(128, 3, activation='relu')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

residual connections when max pooling layers are there.

inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, activation='relu')(inputs)
residual = x
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
x = layers.MaxPooling2D(2, padding='same')(x)  # note that in max_pooling layer also we need to add padding='same'
residual = layers.Conv2D(64, 1, strides=2)(residual)
x = layers.add([x, residual])
outputs = layers.Conv2D(128, 3, activation='relu')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

inputs = keras.Input(shape=(32, 32, 3))
x = layers.Rescaling(1. / 255)(inputs)


def residual_block(x, filters, pooling=False):
    """

    :param x:
    :param filters:
    :param pooling:
    :return:
    """
    residual = x
    x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)
    if pooling:
        x = layers.MaxPooling2D(2, padding='same')(x)
        residual = layers.Conv2D(filters, 1, strides=2)(residual)
    elif filters != residual.shape[-1]:
        residual = layers.Conv2D(filters, 1)(residual)
    x = layers.add([x, residual])
    return x


x = residual_block(x, filters=32, pooling=True)
x = residual_block(x, filters=64, pooling=True)
x = residual_block(x, filters=128, pooling=False)

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

