import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# loading mnist data
(train_images, train_labels), _ = mnist.load_data()

# reshaping the training data
train_images = train_images.reshape(60000, 28 *  28)
train_images = train_images.astype('float32') / 255

random_train_labels = train_labels[:]
np.random.shuffle(random_train_labels)

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
model.fit(x=train_images, y=random_train_labels,
          batch_size=128,
          epochs=10,
          validation_split=0.2)