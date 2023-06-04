import re
import tensorflow as tf

def message_length(x):
    """
    returns the number of characters in the text x (input).
    :param x:
    :return:
    """
    return len(x)

def num_capitals(x):
    """
    returns the number of capital  in the text x (input).
    :param x:
    :return:
    """
    _, count = re.subn(r'[A-Z]', '', x)
    return count

def num_punctuation(x):
    """
    returns the number of characters in the text x (input).
    :param x:
    :return:
    """
    _, count = re.subn(r'\W', '', x)
    return count


def make_model(input_dims=3, num_units=12):
    """

    :param input_dims:
    :param num_units:
    :return:
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(num_units,
                                    input_dim=input_dims,
                                    activation='relu'))
    model.add(tf.keras.layers.Dense(1,
                                    activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
