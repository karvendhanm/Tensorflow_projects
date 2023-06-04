import io
import os
import pandas as pd
import subprocess
import tensorflow as tf

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

