import matplotlib.pyplot as plt
import os

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
