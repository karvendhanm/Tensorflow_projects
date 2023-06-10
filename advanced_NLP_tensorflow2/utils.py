import pandas as pd
import re
import stopwordsiso as stopwords
import stanza as snlp
import tensorflow as tf

# tokenize processor from stanfordnlp
en = snlp.Pipeline(lang='en', download_method=None, processors=['tokenize', 'pos'])

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

def word_counts(x, pipeline=en):
    """

    :param x:
    :param pipeline:
    :return:
    """
    count = 0
    doc = pipeline(x)
    for sentence in doc.sentences:
        for token in sentence.tokens:
            if token.text.lower() not in stopwords.stopwords('en'):
                count += 1
    return count


def word_counts_v2(x, pipeline=en):
    """

    :param x:
    :param pipeline:
    :return:
    """
    doc = pipeline(x)
    tot_word_count = 0
    no_punct = 0
    punct = 0
    for sentence in doc.sentences:
        tot_word_count += len(sentence.tokens)
        for token in sentence.tokens:
            if token.text.lower() not in stopwords.stopwords('en'):
                if token.words[0].upos not in ['PUNCT', 'SYM']:
                    no_punct += 1
                else:
                    punct += 1
    punct = punct/tot_word_count
    return pd.Series([no_punct, punct], index=['Words_NoPunct', 'Punct'])
























