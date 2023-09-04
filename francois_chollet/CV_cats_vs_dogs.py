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

        if not os.path.exists(new_base_dir / subset_name / category): # using / cool feature of pathlib
            pathlib.Path.mkdir(new_base_dir / subset_name / category, parents=True, exist_ok=True)

        fnames = [f'{category}.{i}.jpg' for i in range(start_index, stop_index)]

        for fname in fnames:
            shutil.copy(old_dir / fname, new_base_dir / subset_name / category / fname)

    return None


make_subset(subset_name='train', start_index=0, stop_index=1000)
make_subset(subset_name='validation', start_index=1000, stop_index=1500)
make_subset(subset_name='test', start_index=1500, stop_index=2500)

# since this computer vision is overfitting, we will use data augumentation.
# data augumentation is a regularization technique universally used for CV problems.
data_augumentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2)
])

# building the model
inputs = keras.Input(shape=(180, 180, 3))
x = data_augumentation(inputs)
x = layers.Rescaling(1./255)(x)
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu', name='first_conv_layer')(x)
x = layers.MaxPooling2D(pool_size=2, name='first_max_pool_layer')(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation='relu', name='second_conv_layer')(x)
x = layers.MaxPooling2D(pool_size=2, name='second_max_pool_layer')(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation='relu', name='third_conv_layer')(x)
x = layers.MaxPooling2D(pool_size=2, name='third_max_pool_layer')(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation='relu', name='fourth_conv_layer')(x)
x = layers.MaxPooling2D(pool_size=2, name='fourth_max_pool_layer')(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation='relu', name='fifth_conv_layer')(x)
x = layers.Flatten(name='flattening_layer')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid', name='output_layer')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()
keras.utils.plot_model(model, './model_structure/cv_cat_vs_dogs.png', show_shapes=True)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

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

for data_batch, label_batch in train_dataset:
    print('data batch shape:', data_batch.shape)
    print('label batch shape:', label_batch.shape)
    break

# fitting the model:
callback_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='./model_checkpoints/convnet_from_scratch.keras',
        monitor='val_loss',
        save_best_only=True
    )
]

history = model.fit(train_dataset,
                    epochs=100,
                    validation_data=validation_dataset,
                    callbacks=callback_list)
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

test_model = keras.models.load_model('./model_checkpoints/convnet_from_scratch.keras')
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f'the test accuracy is: {test_acc:.3f}')

# the code below is not working
# plt.figure(figsize=(10, 10))
# for images, _ in train_dataset.take(1):
#     for i in range(9):
#         augmented_images = data_augumentation(images)
#         ax = plt.subplot(3, 3, i+1)
#         plt.imshow(augmented_images[0].numpy().astype('uint8'))
#         plt.axis('off')



























