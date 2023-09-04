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

        if not os.path.exists(new_base_dir / subset_name / category): # using / cool feature of pathlib
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

# instantiating the VGG16 convolutional base:
conv_base = keras.applications.vgg16.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(180, 180, 3)
)
