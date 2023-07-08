import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=100000)

# there are a total of 46 classes
print(np.unique(train_labels, return_counts=True))

"""
each sample in the training data is a list of integers and we can't send in a list of integers 
as an input to the neural network. we need to vectorize the sequence/multi-hot encode it.
"""
print(train_data[0])

# preprocessing input data
def vectorize_sequences(inputs, dimension=100000):
    """

    :param inputs:
    :param dimension:
    :return:
    """
    results = np.zeros(shape=(len(inputs), dimension))
    for row, samples in enumerate(inputs):
        for col in samples:
            results[row, col] = 1
    return results

# vectorize training and test data
X_train = vectorize_sequences(train_data)
X_test = vectorize_sequences(test_data)

# preprocessing target labels
"""
y_train = np.asarray(train_labels).astype('float32')

here np.array(), np.asarray(), np.asarray().astype('int32'), np.asarray().astype('float32') were all
tried out and gives the same evaluation metrics save for the difference in randomness of
model initialization weights.
"""
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


"""
this is a multi-class classification problem.
In a multiclass classification problem there are 2 ways to encode the target labels.
1) do categorical encoding/multi-hot encoding of the target variables, 
and use 'categorical_crossentropy' as loss function.
2) cast the target labels as integer tensor and use sparse_categorical_crossentropy as loss function,

Here we will use the second method.
"""

# define the layers
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(46, activation='softmax')
])

# define loss function, optimizer and evaluation metrics
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# lets break a chunk of train data and create a validation set.
X_val = X_train[:1000]
partial_X_train = X_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

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
plt.savefig('./data/images/reuters_sparse_categorical_crossentropy.png')

print(model.evaluate(X_test, y_test))




