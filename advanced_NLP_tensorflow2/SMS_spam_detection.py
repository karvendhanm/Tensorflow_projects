import io
import os
import pandas as pd

import subprocess
import tensorflow as tf

from advanced_NLP_tensorflow2 import utils


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
df['Length'] = df['Message'].apply(utils.message_length)
df['Capitals'] = df['Message'].apply(utils.num_capitals)
df['Punctuation'] = df['Message'].apply(utils.num_punctuation)
df['Punctuation'] = df['Message'].apply(utils.num_punctuation)
df['Words'] = df['Message'].apply(utils.word_counts)
df.describe()

df['Spam'].value_counts(normalize=True)

# building train and test set
train = df.sample(frac=0.8, random_state=42)
train['Spam'].value_counts(normalize=True)

test = df.drop(index=train.index)
test['Spam'].value_counts(normalize=True)

# resetting the indices
train.reset_index(inplace=True, drop=True)
test.reset_index(inplace=True, drop=True)

x_train, y_train = train[['Length', 'Capitals', 'Punctuation', 'Words']], train[['Spam']]
x_test, y_test = test[['Length', 'Capitals', 'Punctuation', 'Words']], test[['Spam']]

# Modeling normalized data
model = utils.make_model(input_dims=4)
model.fit(x_train, y_train, epochs=10, batch_size=10)

model.evaluate(x_test, y_test)

y_train_pred = model.predict(x_train)

# convert the probability of 0.5 or above as spam
mask = y_train_pred >= 0.5
y_train_pred[mask] = 1
y_train_pred[~mask] = 0

tf.math.confusion_matrix(tf.constant(y_train.Spam), y_train_pred)

train.loc[train.Spam == 1].describe()
train.loc[train.Spam == 0].describe()





