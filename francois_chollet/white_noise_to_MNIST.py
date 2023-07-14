import numpy as np

from tensorflow.keras.datasets import mnist

# loading mnist data
(train_images, train_labels), _ = mnist.load_data()

# reshaping the training data
train_images = train_images.reshape(60000, 28 *  28)
train_images = train_images.astype('float32') / 255

# adding white noise to data
train_images_with_noise = np.concatenate([train_images, np.random.random(len(train_images), 28 * 28)], axis=1)
train_images_with_zeros = np.concatenate([train_images, np.zeros(len(train_images), 28 * 28)], axis=1)

# training a model on both these data sets





