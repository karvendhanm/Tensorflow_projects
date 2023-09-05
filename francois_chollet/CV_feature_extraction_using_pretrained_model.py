import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import shutil
import tensorflow as tf

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
            shutil.copy(old_dir / fname, new_base_dir / subset_name / category / fname)

    return None


make_subset(subset_name='train', start_index=0, stop_index=1000)
make_subset(subset_name='validation', start_index=1000, stop_index=1500)
make_subset(subset_name='test', start_index=1500, stop_index=2500)

# data preprocessing.
train_dataset = image_dataset_from_directory(new_base_dir / 'train',
                                             image_size=(180, 180),
                                             batch_size=32)

validation_dataset = image_dataset_from_directory(new_base_dir / 'validation',
                                                  image_size=(180, 180),
                                                  batch_size=32)
test_dataset = image_dataset_from_directory(new_base_dir / 'test',
                                            image_size=(180, 180),
                                            batch_size=32)

# instantiating the VGG16 convolutional base:
conv_base = keras.applications.vgg16.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(180, 180, 3)
)

# visualizing the model
conv_base.summary()


# method-1: Fast feature extraction without data augmentation.
# Sending the data through the convolution's base and store the data in a numpy array.
# then pass the numpy array through a standalone densely connected layer.
def get_features_and_labels(dataset):
    """

    :param dataset:
    :return:
    """
    all_features = []
    all_labels = []

    for images, labels in dataset:
        preprocessed_images = keras.applications.vgg16.preprocess_input(images)
        features = conv_base.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels)

    return np.concatenate(all_features), np.concatenate(all_labels)


train_features, train_labels = get_features_and_labels(train_dataset)
validation_features, validation_labels = get_features_and_labels(validation_dataset)
test_features, test_labels = get_features_and_labels(test_dataset)

print(f'the shape of the output is: {train_features.shape}')

inputs = keras.Input(shape=(5, 5, 512))
x = layers.Flatten()(inputs)
x = layers.Dense(256)(x)  # oddly no activation function has been specified. why?. No non-linearity required?
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
callbacks = [keras.callbacks.ModelCheckpoint(
    filepath='./model_checkpoints/convnet_transfer_learning_feature_extraction.keras',
    monitor='val_loss',
    save_best_only=True)]

history = model.fit(x=train_features,
          y=train_labels,
          epochs=20,
          validation_data=(validation_features, validation_labels),
          callbacks=callbacks)


training_loss = history.history['loss']
training_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']
epochs = range(1, len(training_accuracy) + 1)
plt.plot(epochs, training_accuracy, 'bo', label='Training_accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation_accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, training_loss, 'bo', label='Training_loss')
plt.plot(epochs, val_loss, 'b', label='Validation_loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

test_model = keras.models.load_model('./model_checkpoints/convnet_transfer_learning_feature_extraction.keras')
test_loss, test_acc = test_model.evaluate(test_features, test_labels)
print(f'the test accuracy is: {test_acc:.3f}')

# method-2: extend the convolution base layer to add the densely connect layer at the end/head.
