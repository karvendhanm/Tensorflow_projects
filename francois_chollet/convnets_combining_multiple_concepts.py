import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist

(train_val_images, train_val_labels), (test_images, test_labels) = mnist.load_data()
train_val_images = train_val_images.reshape((len(train_val_images), 28, 28, 1))
train_val_images = train_val_images.astype('float32') / 255
test_images = test_images.reshape(len(test_images), 28, 28, 1)
test_images = test_images.astype('float32') / 255

# creating train and validation set by splitting the train_data

# shuffling the data just to ensure complete randomness
train_val_images = tf.random.shuffle(train_val_images, seed=42)
train_val_labels = tf.random.shuffle(train_val_labels, seed=42)

# splitting the train data to create a validation set.
validation_data = train_val_images[:10000]
train_data = train_val_images[10000:]
validation_labels = train_val_labels[:10000]
train_labels = train_val_labels[10000:]

# convolutional neural networks take tensor of rank 3. (spatial height, spatial width, and channel depth)
# using functional api
inputs = keras.Input(shape=(28, 28, 1))
# with just 1 conv2D layer and 1 maxpool2D layer the test accuracy is 98.17
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
x = layers.MaxPool2D(pool_size=2)(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_data, y=train_labels, batch_size=512, epochs=30, validation_data=(validation_data, validation_labels))

model.evaluate(x=test_images, y=test_labels)
model.metrics_names

ax1, ax2 = plt.subplots(1, 2, figsize=(15, 8))
