"""
here we are going to fine-tune a pretrained model(VGG16).
step 1: import the convolutional base of a pre-trained model.
step 2: freeze the trainable params of the convolutional base.
step 3: add a densely connected layer on top of the convolutional base and add a data_augmentation layer at the bottom.
step 4: train the model.
step 5: noe unfreeze few top layers of the convolutional base and train the model again.
"""
import matplotlib.pyplot as plt
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
            shutil.copyfile(old_dir / fname, new_base_dir / subset_name / category / fname)

    return None


make_subset(subset_name='train', start_index=0, stop_index=1000)
make_subset(subset_name='validation', start_index=1000, stop_index=1500)
make_subset(subset_name='test', start_index=1500, stop_index=2500)

train_dataset = image_dataset_from_directory(new_base_dir / 'train', image_size=(180, 180), batch_size=32)
validation_dataset = image_dataset_from_directory(new_base_dir / 'validation', image_size=(180, 180), batch_size=32)
test_dataset = image_dataset_from_directory(new_base_dir / 'test', image_size=(180, 180), batch_size=32)

# import convolutional base
conv_base = keras.applications.vgg16.VGG16(include_top=False,
                                           weights='imagenet',
                                           input_shape=(180, 180, 3))
conv_base.trainable = False
conv_base.summary()

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
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# train the model before unfreezing the top few layers of the convolutional base from the pretrained model(VGG16).
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

callback_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='./model_checkpoints/before_finetuning_pretrained_model.keras',
        save_best_only=True,
        monitor='val_loss'
    )
]

history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=validation_dataset,
                    callbacks=callback_list)

model = keras.models.load_model('./model_checkpoints/before_finetuning_pretrained_model.keras')
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'the test accuracy before fine tuning the pre-trained model: {test_accuracy}')

model.summary()
model.layers[-5].summary()

# unfreezing the top four layers of the convolutional base of the pre-trained model(VGG16).
# the next 4 lines doesn't seem to work.
conv_base.summary()
conv_base.trainable = True
for layer in conv_base.layers[:-4]:
    layer.trainable = False

model.layers[-5].trainable = True
for model_layer in model.layers[-5].layers[:-4]:
    model_layer.trainable = False
model.summary()

# given that we have unfreezed the top 4 layers of the convolutional base,
# lets compile the model with low learning rate and train the model again
model.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=keras.optimizers.RMSprop(learning_rate=1e-5))

callback_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='./model_checkpoints/after_finetuning_pretrained_model.keras',
        save_best_only=True,
        monitor='val_loss'
    )
]

history = model.fit(train_dataset,
                    epochs=30,
                    validation_data=validation_dataset,
                    callbacks=callback_list)

model = keras.models.load_model('./model_checkpoints/after_finetuning_pretrained_model.keras')
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'the test accuracy after fine tuning the pre-trained model: {test_accuracy}')

model.summary()
model.layers[-5].weights
