import matplotlib.pyplot as plt
import numpy as np
import os
import random

from tensorflow.keras.utils import load_img, img_to_array
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















