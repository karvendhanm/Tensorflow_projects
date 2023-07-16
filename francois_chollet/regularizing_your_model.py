import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers, regularizers
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
original_model_validation_loss = history_original.history['val_loss']
small_model_validation_loss = history_smaller_model.history['val_loss']
epochs = range(1, len(original_model_validation_loss) + 1)

plt.figure(figsize=(7, 5))
plt.plot(epochs, original_model_validation_loss, 'b--', label='validation loss of original model')
plt.plot(epochs, small_model_validation_loss, 'b',label='validation loss of smaller model')
plt.xlabel('epochs')
plt.ylabel('validation loss')
plt.title('validation_loss - bigger vs smaller model')
plt.legend()
plt.savefig('./data/images/validation_loss-original_vs_smaller_model.png')


# building a much bigger model (overkill - model capacity more than what the problem warrants).
def load_overkill_model():
    """

    :return:
    """
    model = keras.Sequential([
        layers.Dense(512, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


overkill_model = load_overkill_model()
history_overkill_model = overkill_model.fit(x=X_train, y=train_labels,
          batch_size=512,
          epochs=20,
          validation_split=0.4)

overkill_model_validation_loss = history_overkill_model.history['val_loss']
plt.figure(figsize=(7, 5))
plt.plot(epochs, overkill_model_validation_loss, 'r', label='validation loss of overkill model')
plt.plot(epochs, original_model_validation_loss, 'b', label='validation loss of original model')
plt.plot(epochs, small_model_validation_loss, 'g',label='validation loss of smaller model')
plt.xlabel('epochs')
plt.ylabel('validation loss')
plt.title('validation_loss - bigger vs smaller model')
plt.legend()
plt.savefig('./data/images/validation_loss-original_vs_smaller_vs_overkill_model.png')

# Adding weight regularization (L1 and L2 regularization)
# model with L2 regularizer
def load_model_with_regularizer():
    """

    :return:
    """
    model = keras.Sequential([
        layers.Dense(16,
                     kernel_regularizer = regularizers.l2(0.002),
                     activation = 'relu'),
        layers.Dense(16,
                     kernel_regularizer = regularizers.l2(0.002),
                     activation = 'relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


regularizer_model = load_model_with_regularizer()
history_regularizer_model = regularizer_model.fit(x=X_train, y=train_labels,
          batch_size=512,
          epochs=20,
          validation_split=0.4)

regularizer_model_loss = history_regularizer_model.history['val_loss']
plt.figure(figsize=(7, 5))
plt.plot(epochs, regularizer_model_loss, 'y', label='validation loss of l2 regularized model')
plt.plot(epochs, overkill_model_validation_loss, 'r', label='validation loss of overkill model')
plt.plot(epochs, original_model_validation_loss, 'b', label='validation loss of original model')
plt.plot(epochs, small_model_validation_loss, 'g',label='validation loss of smaller model')
plt.xlabel('epochs')
plt.ylabel('validation loss')
plt.title('validation_loss - bigger vs smaller model')
plt.legend()
plt.savefig('./data/images/validation_loss-original_vs_regularized.png')


# drop out for generalization
def load_model_with_dropout():
    """

    :return:
    """
    model = keras.Sequential([
        layers.Dense(16, activation = 'relu'),
        layers.Dropout(0.5),
        layers.Dense(16,activation = 'relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

dropout_model = load_model_with_dropout()
history_dropout_model = dropout_model.fit(x=X_train, y=train_labels,
          batch_size=512,
          epochs=20,
          validation_split=0.4)

dropout_model_loss = history_dropout_model.history['val_loss']
plt.figure(figsize=(7, 5))
plt.plot(epochs, regularizer_model_loss, 'y', label='validation loss of l2 regularized model')
plt.plot(epochs, original_model_validation_loss, 'b', label='validation loss of original model')
plt.plot(epochs, dropout_model_loss, 'g',label='validation loss of dropout regularized model')
plt.xlabel('epochs')
plt.ylabel('validation loss')
plt.title('validation_loss - bigger vs smaller model')
plt.legend()
plt.savefig('./data/images/validation_loss-original_vs_regularized_vs_dropout.png')







