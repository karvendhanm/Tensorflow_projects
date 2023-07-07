import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical

matplotlib.use('agg')

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=100000)

print(train_data[0]) # a list of integers

# decoding the message
word_dict = reuters.get_word_index()
reverse_word_dict = {val:word for word, val in word_dict.items()}

def decode_vectors(vector):
    decoded_str = " ".join([reverse_word_dict.get(num - 3, '?') for num in vector])
    return decoded_str

print(decode_vectors(train_data[0]))
train_labels[0]

# vectorizing the predictors
def vectorize_sequences(inputs, dimension=100000):
    """

    :param inputs:
    :param dimension:
    :return:
    """
    results = np.zeros(shape=(len(inputs), dimension))
    for _idx, input_ in enumerate(inputs):
        for elem in input_:
            results[_idx, elem] = 1
    return results

X_train = vectorize_sequences(train_data)
X_test = vectorize_sequences(test_data)

def categorical_encoding(targets, dimension=46):
    """

    :param targets:
    :param dimension:
    :return:
    """
    results = np.zeros(shape=(len(targets), dimension))
    for _idx, target in enumerate(targets):
        results[_idx, target] = 1
    return results

y_train = categorical_encoding(train_labels)
y_test = categorical_encoding(test_labels)

# inbuild keras function for categorical encoding
# y_train_ = to_categorical(train_labels)
# y_test_ = to_categorical(test_labels)

# creating validation set from training data
X_val = X_train[:1000]
partial_X_train = X_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

# model architecture
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(46, activation='softmax')  # this is a multi-class classification with 46 labels
])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x=partial_X_train, y=partial_y_train, batch_size=512, epochs=20, validation_data=(X_val, y_val))

# loss and accuracy plots (comparing training and validation sets)
history_dict = history.history
training_loss = history_dict['loss']
validation_loss = history_dict['val_loss']
training_accuracy = history_dict['accuracy']
validation_accuracy = history_dict['val_accuracy']
epochs = range(1, len(training_loss) + 1)

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes = axes.flatten()
axes[0].plot(epochs, training_loss, 'bo', label='Training loss')
axes[0].plot(epochs, validation_loss, 'b', label='Validation loss')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].set_xticks(epochs)
axes[0].grid()
axes[0].set_title('training loss vs validation loss')
axes[0].legend()
axes[1].plot(epochs, training_accuracy, 'bo', label='Training acc')
axes[1].plot(epochs, validation_accuracy, 'b', label='Validation acc')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('training accuracy vs validation accuracy')
axes[1].legend()
axes[1].grid()
axes[1].set_xticks(epochs)
plt.savefig('./data/images/reuters.png')

# by scrutnizing the plots it looks like the model starts overfitting at epoch 9.
# lets refit the model with 9 epochs.

# model architecture
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(46, activation='softmax')  # this is a multi-class classification with 46 labels
])

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x=partial_X_train, y=partial_y_train, batch_size=512, epochs=9, validation_data=(X_val, y_val))
results = model.evaluate(X_test, y_test)
print(results)

# establishing a baseline by building a random classifier
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = test_labels_copy == test_labels
print(hits_array.mean())

# generating predictions on new data
predictions = model.predict(X_test)
predictions[0].shape
np.sum(predictions[0])

predictions[0]
# the class with the highest probability
np.argmax(predictions[0])


