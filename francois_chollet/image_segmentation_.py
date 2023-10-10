import matplotlib.pyplot as plt
import numpy as np
import os
import random

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
from tensorflow.keras.utils import image_dataset_from_directory

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

input_dir = './data/images/'
target_dir = './data/annotations/trimaps/'

input_img_paths = sorted(
    [os.path.join(input_dir, fname)
     for fname in os.listdir(input_dir)
     if fname.endswith('.jpg')])
target_paths = sorted(
    [os.path.join(target_dir, fname)
     for fname in os.listdir(target_dir)
     if fname.endswith('.png') and not fname.startswith('.')])

"""
if the image is loaded through image dataset from directory,
we can view the files like this
"""
# train_dataset = image_dataset_from_directory('./data/image_segmentation/cats/',
#                                              image_size=(180, 180),
#                                              batch_size=32)
# for image, label in train_dataset.take(1):
#     image_unit8 = image.astype('uint8')
#     plt.imshow(image_unit8[30])
#     plt.axis('off')
#     plt.savefig('./plots/image_segmentation.png')

# another way of looking at pictures is to use load_img from keras.utils
plt.imshow(load_img(input_img_paths[9]))
plt.axis('off')
plt.savefig('./plots/image_segmentation_load_img.png')


# # plot the target image
# plt.imshow(load_img(target_paths[9], color_mode='grayscale'))
# plt.axis('off')
# plt.savefig('./plots/image_segmentation_no_frills.png')

# visualizing targets or segmentation maps
def display_target(target_array):
    """

    :return:
    """
    target_array = (target_array.astype('uint8') - 1) * 127
    plt.imshow(target_array)
    plt.axis('off')
    plt.savefig('./plots/target_segmentation_map_load_img.png')


img = img_to_array(load_img(target_paths[9], color_mode='grayscale'))
display_target(img)

# shuffling both image and target paths to ensure randomness.
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_paths)


def path_to_input_image(input_path):
    """

    :param input_path:
    :return:
    """
    return img_to_array(load_img(input_path, target_size=img_size))


def path_to_target(target_path):
    """

    :return:
    """
    target_array = img_to_array(load_img(target_path, target_size=img_size, color_mode='grayscale'))
    target_array = target_array.astype('uint8') - 1
    return target_array


img_size = (200, 200)
num_imgs = len(input_img_paths)
input_imgs = np.zeros(shape=(num_imgs,) + img_size + (3,), dtype='float32')
targets = np.zeros(shape=(num_imgs,) + img_size + (1,), dtype='uint8')
for i in range(num_imgs):
    input_imgs[i] = path_to_input_image(input_img_paths[i])
    targets[i] = path_to_target(target_paths[i])

num_val_samples = 1000
train_input_imgs = input_imgs[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_input_images = input_imgs[-num_val_samples:]  # don't forget the minus sign out here.
val_targets = targets[-num_val_samples:]


# defining the model.
def get_model(image_size, num_classes):
    """

    :param image_size:
    :param num_classes:
    :return:
    """
    inputs = keras.Input(shape=image_size + (3,))
    x = layers.Rescaling(1. / 255)(inputs)

    x = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(256, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(256, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(256, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
    outputs = layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


model = get_model(img_size, num_classes=3)
model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
callbacks = [
    keras.callbacks.ModelCheckpoint('./model_checkpoints/oxford_segmentation.keras',
                                    save_best_only=True,
                                    monitor='val_loss')
]
# model.fit has been commented as it takes a lot of computational resource.
# history = model.fit(train_input_imgs, train_targets,
#                     validation_data=(val_input_images, val_targets),
#                     epochs=50,
#                     batch_size=64, callbacks=callbacks)

# training_loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(training_loss) + 1)
# plt.figure()
# plt.plot(epochs, training_loss, 'bo', label='training loss')
# plt.plot(epochs, val_loss, 'b', label='validation loss')
# plt.title('training and validation loss')
# plt.legend()
# plt.savefig('./plots/image_segmentation.png')

model = keras.models.load_model('./model_checkpoints/oxford_segmentation.keras')
model.summary()

plt.clf()
i = 4
test_image = val_input_images[i]
plt.imshow(array_to_img(test_image))
plt.axis('off')
plt.savefig('./plots/image_segmentation_test_image.png')

# for prediction, we need to expand the dimension of the image array.
test_image_expanded = np.expand_dims(test_image, axis=0)
pred = model.predict(test_image_expanded)[0]  # reducing the dimension again

plt.clf()
plt.imshow(pred)
plt.axis('off')
plt.savefig('./plots/image_segmentation_predicted_segmentation.png')

def display_segmentation_map(mask):
    """

    :param mask:
    :return:
    """
    mask = np.argmax(mask, axis=-1)
    mask *= 127
    plt.imshow(mask)
    plt.axis('off')
    plt.savefig('./plots/predicted_segmentation_mask.png')



display_segmentation_map(pred)





