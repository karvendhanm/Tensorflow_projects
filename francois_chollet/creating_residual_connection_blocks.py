import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import mnist

"""
# To implement the following with mnist dataset image classification
1) Adding dropout layers
2) Implementing residual connects
3) Implementing batch nomralization
4) Implementing data augmentation
"""

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

