import numpy as np
import operator

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


