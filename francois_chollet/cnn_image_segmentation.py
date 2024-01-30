# pixel level trimap segmentation

import matplotlib.pyplot as plt
import numpy as np
import os
import random

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img, img_to_array, array_to_img

input_dir = './data/images/'
target_dir = './data/annotations/trimaps'

input_img_paths = sorted([os.path.join(input_dir, fname)
                          for fname in os.listdir(input_dir)
                          if fname.endswith(".jpg")])
target_paths = sorted([os.path.join(target_dir, fname)
                       for fname in os.listdir(target_dir)
                       if fname.endswith('.png') and not fname.startswith('.')])


# plt.axis('off')
# plt.imshow(load_img(input_img_paths[9]))
# plt.savefig('./plots/image_segmentation_load_img.png')


def display_target(target_arr):
    """

    :param target_arr:
    :return:
    """
    normalized_array = (target_arr.astype('uint8') - 1) * 127
    plt.axis('off')
    plt.imshow(normalized_array[:, :, 0])
    plt.savefig('./plots/image_segmentation_target_img.png')
    return None


target_array = img_to_array(load_img(target_paths[9], color_mode='grayscale'))
# display_target(target_array)

random.Random(42).shuffle(input_img_paths)
random.Random(42).shuffle(target_paths)

img_size = (200, 200)
num_imgs = len(input_img_paths)


def path_to_input_img(path):
    """

    :param path:
    :return:
    """
    return img_to_array(load_img(path, target_size=img_size))


def path_to_target(path):
    """

    :param path:
    :return:
    """
    img = img_to_array(load_img(path, grayscale=True, target_size=img_size))
    img = img.astype('uint8') - 1
    return img


input_imgs = np.zeros((num_imgs,) + img_size + (3,),
                      dtype='float32')  # interesting property of tuples, number 3 represents the color channels
targets = np.zeros((num_imgs,) + img_size + (1,), dtype='uint8')  # target has only one color channel

for i in range(num_imgs):
    input_imgs[i] = path_to_input_img(input_img_paths[i])
    targets[i] = path_to_target(target_paths[i])

num_val_samples = 1000
train_input_imgs = input_imgs[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_input_imgs = input_imgs[-num_val_samples:]
val_targets = targets[-num_val_samples:]


def get_model(img_size, num_classes):
    """

    :param img_size:
    :param num_classes:
    :return:
    """
    inputs = keras.Input(shape=img_size + (3,))
    x = layers.Rescaling(1. / 255)(inputs)

    x = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(256, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(256, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(256, 3, activation='relu', padding='same', strides=2)(x)
    x = layers.Conv2DTranspose(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(128, 3, activation='relu', padding='same', strides=2)(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same', strides=2)(x)

    outputs = layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)
    model = keras.Model(inputs, outputs)
    return model


model = get_model(img_size, num_classes=3)
model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
callbacks = [
    keras.callbacks.ModelCheckpoint("./model_checkpoints/oxford_segmentation.keras", save_best_only=True)
]
history = model.fit(train_input_imgs, train_targets, epochs=50, callbacks=callbacks,
                    batch_size=64, validation_data=(val_input_imgs, val_targets))

epochs = range(1, len(history.history['loss']) + 1)
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure()
plt.plot(epochs, loss, 'bo', label='training_loss')
plt.plot(epochs, val_loss, 'b', label='validation_loss')
plt.title('training and validation loss')
plt.legend()

model = keras.models.load_model('./model_checkpoints/oxford_segmentation.keras')

i = 4
test_image = val_input_imgs[i]
plt.axis('off')
plt.imshow(array_to_img(test_image))
plt.savefig('./plots/test_image.png')

y_pred = model.predict(np.expand_dims(test_image, axis=0))[0]
mask = np.argmax(y_pred, axis=-1)
mask *= 127
plt.axis('off')
plt.imshow(mask)
plt.savefig('./plots/predicted_image_segment.png')




