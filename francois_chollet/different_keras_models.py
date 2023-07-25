"""
There are 3 different ways to building keras models.
    1) The Sequential model - mostly used by Novice users
    2) Functional API - mostly used by engineers
    3) model subclassing - mostly used by researchers
"""

# the Sequential model/sequential API example

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation='relu', name='first_layer'),
    layers.Dense(10, activation='softmax', name='second_layer')
], name='my sequential model')

model.weights

# building the model
model.build(input_shape=(None, 3))

model.weights

model.summary()

# building the same sequential model incrementally
model_ = keras.Sequential()
model_.add(layers.Dense(64, activation='relu'))
model_.add(layers.Dense(10, activation='softmax'))

model_.build(input_shape=(None, 3))

model_.weights

model_.summary()

# building model on the fly
model_v1 = keras.Sequential(name='sequential_model')
model_v1.add(keras.Input(shape=(3,)))
model_v1.add(layers.Dense(64, activation='relu'))
model_v1.summary()
model_v1.weights

# adding the next layer
model_v1.add(layers.Dense(10, activation='softmax'))
model_v1.weights
model_v1.summary()








