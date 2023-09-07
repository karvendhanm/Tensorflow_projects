"""
Fine-tuning a pre-trained model.

Here a densely connected classifier is added on top of the pre-trained model(VGG16, a convolutional base).

then entire convolutional base is freezed, and the added densely connected classifier is trained.

then after the densely connected classifier is trained, few top layers (specialized layers) of the

convolutional base is unfreezed, and we again train both the newly unfrozen layers in the convolutional base

and the densely connected classifier.

"""
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

# method-1: Fast feature extraction without data augmentation.
# code for this method is present in "CV_feature_extraction_using_pretrained_model.py".

# method-2: extend the convolution base layer to add the densely connect layer at the end/head.
# Feature extraction with data augmentation.

# instantiating the VGG16 convolutional base:
conv_base = keras.applications.vgg16.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(180, 180, 3)
)
# need to freeze the weights of the convolution base of the VGG16.
conv_base.trainable = False

conv_base.summary()     # note that the trainable parameters for convolutional base is zero.

"""
Now we can create a new model that chains together 
    1) A data augmentation stage.
    2) Our frozen convolutional base.
    3) A dense classifier.
"""

data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
])

inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = keras.applications.vgg16.preprocess_input(x)
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

conv_base.summary()
model.summary()

# freezing top 3 convolution layers and a max pooling layer.
conv_base.trainable = True
for layer in conv_base.layers[:-4]:
    layer.trainable = False

conv_base.summary()
model.summary()

# we will have a very small learning rate, as large updates might change the weights
# of the pre-trained model too much even as we are training the model with a very small dataset,
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])

callbacks = [keras.callbacks.ModelCheckpoint(
    filepath='./model_checkpoints/convnet_transfer_learning_fine_tuning.keras',
    monitor='val_loss',
    save_best_only=True)]

history = model.fit(train_dataset,
          epochs=30,
          validation_data=validation_dataset,
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

test_model = keras.models.load_model('./model_checkpoints/convnet_transfer_learning_fine_tuning.keras')
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f'the test accuracy is: {test_acc:.3f}')





























