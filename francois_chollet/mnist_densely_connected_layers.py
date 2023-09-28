# classifying digits of mnist dataset using dense connected neural networks

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

train_data = train_data.reshape(60000, 28 * 28).astype('float32') / 255
test_data = test_data.reshape(10000, 28 * 28).astype('float32') / 255

# getting validation data
validation_data = train_data[:10000]
validation_labels = train_labels[:10000]
reduced_train_data = train_data[10000:]
reduced_train_labels = train_labels[10000:]

inputs = keras.Input(shape=(28 * 28,), name='input_layer')
x = layers.Dense(512, activation='relu', name='first_dense_layer')(inputs)
outputs = layers.Dense(10, activation='softmax', name='output_layer')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name='my_model')

model.summary()
tf.keras.utils.plot_model(model, to_file='model_structure/mnist.png', show_shapes=True)

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='./model_checkpoints/mnist/dense_connected_layers.keras',
                                       monitor='val_loss',
                                       save_best_only=True),
]

model.fit(
    x=reduced_train_data,
    y=reduced_train_labels,
    callbacks=my_callbacks,
    batch_size=128,
    epochs=30,
    validation_data=(validation_data, validation_labels),
    shuffle=True
)

model = keras.models.load_model('./model_checkpoints/mnist/dense_connected_layers.keras')
test_loss, test_accuracy = model.evaluate(x=test_data, y=test_labels)



