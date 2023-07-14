import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

matplotlib.use('agg')

# loading mnist data
(train_images, train_labels), _ = mnist.load_data()

# reshaping the training data
train_images = train_images.reshape(60000, 28 *  28)
train_images = train_images.astype('float32') / 255

# adding white noise to data
train_images_with_noise = np.concatenate([train_images, np.random.random((len(train_images), 28 * 28))], axis=1)
train_images_with_zeros = np.concatenate([train_images, np.zeros((len(train_images), 28 * 28))], axis=1)

# training a model on both these data sets (with white noise and with zeros)

def get_model():
    """
    model architecture and picking optimizer, loss functon and metrics
    :return:
    """
    model = keras.Sequential([
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy', # used for integer labels
                  metrics=['accuracy']
                  )

    return model


model = get_model()
history_noise = model.fit(x=train_images_with_noise, y=train_labels,
                          batch_size=128,
                          epochs=10,
                          validation_split=0.2)

history_zeros = model.fit(x=train_images_with_zeros, y=train_labels,
                          batch_size=128,
                          epochs=10,
                          validation_split=0.2)

# plotting the progress of validation accuracy with epochs
noise_accuracy = history_noise.history['val_accuracy']
zeros_accuracy = history_zeros.history['val_accuracy']
epochs = range(1, len(noise_accuracy) +  1)

plt.figure(figsize = (5, 7))
plt.plot(epochs, noise_accuracy, 'b-', label='validation accuracy with noise channels')
plt.plot(epochs, zeros_accuracy, 'b--',  label='validation accuracy with zeros channels')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('./data/images/white_noise_on_MNIST.png')















