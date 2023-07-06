import matplotlib.pyplot as plt
import numpy as np
import operator

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

for i in [0, 234, 43, 5146, 9000, 24003]:
    sample = train_data[i]
    print(sample)
    print(type(sample))
    print(f'the length of the sample data is: {len(sample)}')

word_index = imdb.get_word_index()

reverse_word_index = {value: key for key, value in word_index.items()}
# temp = sorted(reverse_word_index.items(), key=operator.itemgetter(0), reverse=False)

# we are using i-3 as 0, 1, 2 are reserved for 'padding', 'start of sequence', and 'unknown'
decoded_review = " ".join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_review)

# preparing the data:
# encoding the integer sequence via multi-hot encoding
def vectorize_sequences(sequences, dimension=10000):
    """

    :param sequences:
    :param dimension:
    :return:
    """
    results = np.zeros(shape=(len(sequences), dimension))
    for row, sequence in enumerate(sequences):
        for col in sequence:
            results[row, col] = 1
    return results
X_train = vectorize_sequences(train_data)
X_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# build the model
model = keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# choose loss function, optimizer and metrics
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

X_val = X_train[:10000]
partial_X_train = X_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(x=partial_X_train, y=partial_y_train, batch_size=512,
                    epochs=20, validation_data=(X_val, y_val))

# history object has a member history containing data about everything that happened
# during training
# history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# epochs = range(1, len(loss_values)+1)
# plt.plot(epochs, loss_values, "bo", label='Training loss')
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# plt.clf() # clears the figure
# acc = history_dict['accuracy']
# val_acc = history_dict['val_accuracy']
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and Validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# the model seems to overfit (over optimization on the training data) after 4th epoch.
# lets train the same model again with only 4 epochs
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x=partial_X_train, y=partial_y_train, batch_size=512, epochs=4)
results = model.evaluate(X_test, y_test)
results













