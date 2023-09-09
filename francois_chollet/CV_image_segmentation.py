import matplotlib.pyplot as plt
import numpy as np
import os
import random

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img, img_to_array, array_to_img

input_dir = 'data/images/'
target_dir = 'data/annotations/trimaps/'

input_image_paths = sorted([os.path.join(input_dir, fname)
                            for fname in os.listdir(input_dir)
                            if fname.endswith('.jpg')])

target_paths = sorted([os.path.join(target_dir, fname)
                       for fname in os.listdir(target_dir)
                       if fname.endswith('.png') and not fname.startswith('.')])

plt.axis('off')
plt.imshow(load_img(input_image_paths[9]))
plt.show()


def display_target(target_array):
    """

    :param target_array:
    :return:
    """
    normalized_array = (target_array.astype('uint8') - 1) * 127
    plt.axis('off')
    plt.imshow(normalized_array)
    plt.show()


img_gray = img_to_array(load_img(target_paths[9], color_mode='grayscale'))
display_target(img_gray)

# training and validation sets
img_size = (200, 200)
num_images = len(input_image_paths)

random.Random(1331).shuffle(input_image_paths)
random.Random(1331).shuffle(target_paths)


def path_to_input_image(path):
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
    img = img_to_array(load_img(path, target_size=img_size, color_mode='grayscale'))
    img = img.astype('uint8') - 1
    return img


input_imgs = np.zeros(shape=(num_images,) + img_size + (3,), dtype='float32')
targets = np.zeros(shape=(num_images,) + img_size + (1,), dtype='uint8')
for i in range(num_images):
    input_imgs[i] = path_to_input_image(input_image_paths[i])
    targets[i] = path_to_target(target_paths[i])

num_val_samples = 1000
train_input_imgs = input_imgs[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_input_images = input_imgs[-num_val_samples:]
val_targets = targets[-num_val_samples:]


# define the model
def get_model(img_size, num_classes):
    """

    :param img_size:
    :param num_classes:
    :return:
    """

    # input and rescaling layer
    inputs = keras.Input(shape=img_size + (3,), name='input_layer')
    x = layers.Rescaling(1. / 255, name='rescaling_layer')(inputs)

    # conv2d layers with and without strides
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='same', name='conv2d_first')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same', name='conv2d_second')(x)
    x = layers.Conv2D(128, 3, strides=2, activation='relu', padding='same', name='conv2d_third')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same', name='conv2d_fourth')(x)
    x = layers.Conv2D(256, 3, strides=2, activation='relu', padding='same', name='conv2d_fifth')(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same', name='conv2d_sixth')(x)

    # conv2d transpose layers without and with strides
    x = layers.Conv2DTranspose(256, 3, activation='relu', padding='same', name='conv2d_transpose_first')(x)
    x = layers.Conv2DTranspose(256, 3, activation='relu', padding='same', strides=2, name='conv2d_transpose_second')(x)
    x = layers.Conv2DTranspose(128, 3, activation='relu', padding='same', name='conv2d_transpose_third')(x)
    x = layers.Conv2DTranspose(128, 3, activation='relu', padding='same', strides=2, name='conv2d_transpose_fourth')(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same', name='conv2d_transpose_fifth')(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same', strides=2, name='conv2d_transpose_sixth')(x)

    # output layer
    outputs = layers.Conv2D(num_classes, 3, activation='softmax', padding='same', name='output_layer')(x)
    model = keras.Model(inputs, outputs)
    return model


model = get_model(img_size=img_size, num_classes=3)
model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
callbacks = [
    keras.callbacks.ModelCheckpoint('oxford_segmentation.keras', save_best_only=True)
]

history = model.fit(train_input_imgs, train_targets,
                    epochs=50,
                    callbacks=callbacks,
                    batch_size=64,
                    validation_data=(val_input_images, val_targets))

training_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(training_loss) + 1)
plt.plot(epochs, training_loss, 'bo', label='Training_loss')
plt.plot(epochs, val_loss, 'b', label='Validation_loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

model = keras.models.load_model('oxford_segmentation.keras')

i = 4
test_image = val_input_images[i]
plt.axis('off')
plt.imshow(array_to_img(test_image))
plt.show()

mask = model.predict(np.expand_dims(test_image, 0))[0]

def display_mask(pred):
    """

    :param pred:
    :return:
    """
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.axis("off")
    plt.imshow(mask)
    plt.show()

display_mask(mask)






























