"""
As far as computer vision goes,
data augmentation is essential to prevent over fitting in small datasets.
"""

import os
import pathlib
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import keras.applications.vgg16
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

old_dir = pathlib.Path('./data/train')
new_base_dir = pathlib.Path('./data/cats_vs_dogs_small')


def make_subset(subset_name, start_index, stop_index):
    """

    :param subset_name:
    :param start_index:
    :param stop_index:
    :return:
    """
    for category in ('cat', 'dog'):

        if not os.path.exists(new_base_dir / subset_name / category):  # using / cool feature of pathlib
            pathlib.Path.mkdir(new_base_dir / subset_name / category, parents=True, exist_ok=True)

        fnames = [f'{category}.{i}.jpg' for i in range(start_index, stop_index)]

        for fname in fnames:
            shutil.copyfile(old_dir / fname, new_base_dir / subset_name / category / fname)

    return None


make_subset(subset_name='train', start_index=0, stop_index=1000)
make_subset(subset_name='validation', start_index=1000, stop_index=1500)
make_subset(subset_name='test', start_index=1500, stop_index=2500)

train_dataset = image_dataset_from_directory(new_base_dir / 'train', image_size=(180, 180), batch_size=32)
validation_dataset = image_dataset_from_directory(new_base_dir / 'validation', image_size=(180, 180), batch_size=32)
test_dataset = image_dataset_from_directory(new_base_dir / 'test', image_size=(180, 180), batch_size=32)

##### START EXPERIMENT
"""
this experiment, for 30 epochs, resulted in a training accuracy of 0.7042 and validation accuracy of 0.6270.
more epochs definitely will result in better accuracy.
however this computationally expensive taking as much as 22s per epoch on a RTX 3060 12GB variant.
"""
#
#
# # import the convolutional base of VGG16 model trained on imageNet.
# # here we are using the densely connected classifier as well.
# model = keras.applications.vgg16.VGG16(include_top=True,
#                                            weights='imagenet',
#                                            input_shape=(224, 224, 3))
# model.summary()
#
# # compilation step
# model.compile(optimizer='rmsprop',
#               loss='sparse_categorical_crossentropy',
#               metrics='accuracy')
#
# history = model.fit(train_dataset, epochs=30, validation_data=validation_dataset)
##### STOP EXPERIMENT

# using a pre-trained VGG16 architecture but without the top_layer.
# in other words using only the convolutional base and discarding the densely connected classifier at the top.
# convolutional base consists of conv2d and maxpooling layers.
conv_base = keras.applications.vgg16.VGG16(include_top=False,
                                           weights='imagenet',
                                           input_shape=(180, 180, 3))
conv_base.summary()


def get_features_and_labels(dataset):
    """

    :param dataset:
    :return:
    """
    all_features = []
    all_labels = []
    for images, labels in dataset:
        preprocessed_images = keras.applications.vgg16.preprocess_input(images)
        # let us extract interesting features from the images using the representations
        # learned by the pre-trained model VGG16.
        features = conv_base.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels)
    return np.concatenate(all_features), np.concatenate(all_labels)


train_features, train_labels = get_features_and_labels(train_dataset)
validation_features, validation_labels = get_features_and_labels(validation_dataset)
test_features, test_labels = get_features_and_labels(test_dataset)

# creating a densely connected classifier.
inputs = keras.Input(shape=(5, 5, 512))
x = layers.Flatten()(inputs)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# compile the model
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='./model_checkpoints/conv_base_with_newly_trained_densely_connected_classifier.keras',
        save_best_only=True,
        monitor='val_loss'
    )
]

history = model.fit(x=train_features,
          y=train_labels,
          epochs=20,
          validation_data=(validation_features, validation_labels),
          batch_size=32,
          callbacks=callbacks_list)

training_loss = history.history['loss']
training_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']
epochs = list(range(1, len(training_loss) + 1))

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(epochs, training_loss, 'bo', label="training loss")
ax1.plot(epochs, val_loss, 'b', label="validation loss")
ax1.set_title('training loss vs. validation loss')
ax1.legend()
ax2.plot(epochs, training_accuracy, 'bo', label="training accuracy")
ax2.plot(epochs, val_accuracy, 'b', label="validation accuracy")
ax2.set_title('training accuracy vs. validation accuracy')
ax2.legend()
plt.savefig('./plots/pretrained_model_with_dense_classifier_built_from_scratch.png')

test_loss, test_accuracy = model.evaluate(x=test_features, y=test_labels,
                                          batch_size=32)
print(f'the test accuracy is: {test_accuracy}')














