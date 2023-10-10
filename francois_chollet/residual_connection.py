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
