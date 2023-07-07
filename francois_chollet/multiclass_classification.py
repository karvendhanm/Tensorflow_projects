import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import reuters
from tensorflow.keras.utils import to_categorical

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






