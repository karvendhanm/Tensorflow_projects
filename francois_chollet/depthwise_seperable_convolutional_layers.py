import matplotlib.pyplot as plt
import os
import pathlib
import shutil
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

# Implement depthwise seperable convolutional neural networks.
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

# defining a data augmentation layer using sequential API
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
])

inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x = layers.Conv2D(filters=32, kernel_size=5, use_bias=False)(x)

for size in (32, 64, 128, 256, 512):
    residual = x

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(size, 3, padding='same', use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(size, 3, padding='same', use_bias=False)(x)

    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    residual = layers.Conv2D(filters=size, kernel_size=1, strides=2, padding='same', use_bias=False)(residual)
    x = layers.add([x, residual])

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='./model_checkpoints/image_classification_using_depthwise_seperable_conv_layers.keras',
        save_best_only=True,
        monitor='val_loss')
]

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_dataset, validation_data=validation_dataset, epochs=30, callbacks=callbacks)

train_accuracy = history.history['accuracy']
train_loss = history.history['loss']
validation_accuracy = history.history['val_accuracy']
validation_loss = history.history['val_loss']
epochs = list(range(1, len(train_accuracy)+1))

# plotting the train accuracy
figs, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
ax1.plot(epochs, train_loss, 'bo', label='training_loss')
ax1.plot(epochs, validation_loss, 'b', label='validation_loss')
ax1.set_title('training loss vs. validation loss')
ax1.grid(True)
ax1.legend()
ax2.plot(epochs, train_accuracy, 'bo', label='training_accuracy')
ax2.plot(epochs, validation_accuracy, 'b', label='validation_accuracy')
ax2.set_title('training accuracy vs. validation accuracy')
ax2.grid(True)
ax2.legend()
plt.savefig('./plots/image_classification_using_depthwise_seperable_conv_layers.png')

best_model = keras.models.load_model('./model_checkpoints/'
                                     'image_classification_using_depthwise_seperable_conv_layers.keras')
test_loss, test_accuracy = best_model.evaluate(test_dataset)
print(f'the test accuracy: {test_accuracy:.3f}')

































