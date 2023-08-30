# In this file mnist dataset has been modelled using CNN(convolutional neural networks)"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

inputs = keras.Input(shape=(28, 28, 1), name='input_layer')  # shape - (image height, image width, image channels)
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu', name='first_conv_layer')(inputs)
x = layers.MaxPooling2D(pool_size=2, name='first_max_pool_layer')(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation='relu', name='second_conv_layer')(x)
x = layers.MaxPooling2D(pool_size=2, name='second_max_pool_layer')(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation='relu', name='third_conv_layer')(x)
x = layers.Flatten(name='flattening_layer')(x)
outputs = layers.Dense(10, activation='softmax', name='output_layer')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

keras.utils.plot_model(model, './model_structure/mnist_cnn.png', show_shapes=True)
model.summary()

# importing mnist data
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data = train_data.reshape(60000, 28, 28, 1).astype('float32') / 255
test_data = test_data.reshape(10000, 28, 28, 1).astype('float32') / 255

model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
model.fit(x=train_data, y=train_labels, epochs=30, batch_size=2048)
