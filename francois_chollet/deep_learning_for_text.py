import string


class Vectorizer(object):
    def __init__(self):
        self.inverse_dictionary = None

    def standardize(self, text):
        text = text.lower()
        return "".join(char for char in text if char not in string.punctuation)

    def tokenize(self, text):
        text = self.standardize(text)
        return text.split()

    def make_vocabulary(self, dataset):
        self.vocabulary = {"": 0, "[UNK]": 1}
        for text in dataset:
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
        self.inverse_dictionary = {v: k for k, v in self.vocabulary.items()}

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, 1) for token in tokens]

    def decode(self, int_sequence):
        return " ".join([self.inverse_dictionary.get(i, '[UNK]') for i in int_sequence])

dataset = [
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy blooms.",
]

obj = Vectorizer()
obj.make_vocabulary(dataset)

test_sentence = "I write, rewrite, and still write again"
encoded_sequence = obj.encode(test_sentence)
print(obj.decode(encoded_sequence))

# text vectorization uisng keras
from tensorflow.keras.layers import TextVectorization
text_vectorization = TextVectorization(output_mode='int')
