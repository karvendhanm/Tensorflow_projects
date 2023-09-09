import matplotlib.pyplot as plt
import os

from tensorflow.keras.utils import load_img, img_to_array

input_dir = 'data/images/'
target_dir = 'data/annotations/trimaps/'

input_image_paths = sorted([os.path.join(input_dir, fname)
                            for fname in os.listdir(input_dir)
                            if fname.endswith('.jpg')])

target_paths = sorted([os.path.join(target_dir, fname)
                       for fname in os.listdir(target_dir)
                       if fname.endswith('.png') and not fname.startswith('.')])

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






