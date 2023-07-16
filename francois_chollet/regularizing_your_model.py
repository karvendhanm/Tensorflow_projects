import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb

matplotlib.use('agg')

(train_data, train_labels), _ = imdb.load_data(num_words=10000)
train_labels = train_labels.astype('int32')


# vectorizing the training data
def vectorize_sequences(inputs, dimension=10000):
    """

    :param inputs:
    :param dimension:
    :return:
    """
    results = np.zeros(shape=(len(inputs), dimension))
    for row, input_ in enumerate(inputs):
        results[row, input_] = 1. # filling up all the columns at once
    return results


X_train = vectorize_sequences(train_data)


def load_model():
    """

    :return:
    """
    model = keras.Sequential([
        layers.Dense(16, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


model = load_model()
history_original = model.fit(x=X_train, y=train_labels,
          batch_size=512,
          epochs=20,
          validation_split=0.4)

# now replace the original model with a smaller model.


def load_smaller_model():
    """

    :return:
    """
    model = keras.Sequential([
        layers.Dense(4, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


small_model=load_smaller_model()
history_smaller_model = small_model.fit(x=X_train, y=train_labels,
          batch_size=512,
          epochs=20,
          validation_split=0.4)

# comparing the validation loss of original model and the smaller model.
big_model_validation_loss = history_original.history['val_loss']2
small_model_validation_loss = history_smaller_model.history['val_loss']
epochs = range(1, len(big_model_validation_loss) + 1)

plt.figure(figsize=(7, 5))
plt.plot(epochs, big_model_validation_loss, 'bo', label='validation loss of original model')
plt.plot(epochs, small_model_validation_loss, 'b',label='validation loss of smaller model')
plt.xlabel('epochs')
plt.ylabel('validation loss')
plt.title('validation_loss - bigger vs smaller model')
plt.legend()
plt.savefig('./data/images/validation_loss-bigger_vs_smaller_model.png')











