# Computer vision - Cats vs. Dogs
import matplotlib.pyplot as plt
import numpy
import os
import pathlib
import shutil

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

# As our data size is small, our model is prone to overfitting
# data augmentation layers
data_augmentation = keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ])

# building a convolutional neural network using functional API
inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

train_dataset = image_dataset_from_directory(new_base_dir / 'train', image_size=(180, 180), batch_size=32)
validation_dataset = image_dataset_from_directory(new_base_dir / 'validation', image_size=(180, 180), batch_size=32)
test_dataset = image_dataset_from_directory(new_base_dir / 'test', image_size=(180, 180), batch_size=32)

# let us fit the model on our dataset.
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='./model_checkpoints/convnet_from_scratch_with_augmentation.keras',
        save_best_only=True,
        monitor='val_loss')
]

history = model.fit(train_dataset,
                    epochs=100,
                    validation_data=validation_dataset,
                    callbacks=callbacks)

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
plt.savefig('./plots/CV_cats_and_dogs_with_data_augmentation.png')

test_model = keras.models.load_model('./model_checkpoints/convnet_from_scratch_with_augmentation.keras')
test_loss, test_accuracy = test_model.evaluate(test_dataset)
print(f'the test accuracy: {test_accuracy:.3f}')

















