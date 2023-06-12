import io
import os
import pandas as pd

import subprocess
import tensorflow as tf

from advanced_NLP_tensorflow2 import utils
from sklearn.feature_extraction.text import TfidfVectorizer

if not os.path.exists('./NLP_data/SMSSpamCollection'):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    path_to_zip = tf.keras.utils.get_file('smsspamcollection.zip', origin=url)

    subprocess.run(['unzip', path_to_zip, '-d', 'NLP_data'])

with io.open('./NLP_data/SMSSpamCollection') as fh:
    lines = fh.read().strip().split('\n')

spam_dataset = []
for line in lines:
    label, text = line.split('\t')
    if label.strip() == 'spam':
        spam_dataset.append((1, text.strip()))
    else:
        spam_dataset.append((0, text.strip()))

df = pd.DataFrame(spam_dataset, columns=['Spam', 'Message'])

# building train and test set
train = df.sample(frac=0.8, random_state=42)
train['Spam'].value_counts(normalize=True)

test = df.drop(index=train.index)
test['Spam'].value_counts(normalize=True)

# embedding the Message column using tfidf vectorizer
vectorizer = TfidfVectorizer(binary=True)
X_train = vectorizer.fit_transform(train['Message']).astype('float32')
X_test = vectorizer.transform(test['Message']).astype('float32')

y_train = train['Spam']
y_test = test['Spam']

# Modeling tfidf vectorizer  data
model = utils.make_model(input_dims=X_train.shape[1])
model.fit(X_train.toarray(), y_train, epochs=10, batch_size=10)

# evaluating the model
test_loss, test_accuracy = model.evaluate(X_test.toarray(), y_test)

y_test_pred = model.predict(X_test.toarray())

# convert the probability of 0.5 or above as spam
mask = y_test_pred >= 0.5
y_test_pred[mask] = 1
y_test_pred[~mask] = 0

tf.math.confusion_matrix(tf.constant(test.Spam), y_test_pred)
