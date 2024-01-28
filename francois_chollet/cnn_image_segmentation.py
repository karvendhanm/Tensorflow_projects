# pixel level trimap segmentation

import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.keras.utils import load_img, img_to_array

input_dir = './data/images/'
target_dir = './data/annotations/trimaps'

input_img_paths = sorted([os.path.join(input_dir, fname)
                          for fname in os.listdir(input_dir)
                          if fname.endswith(".jpg")])
target_paths = sorted([os.path.join(target_dir, fname)
                       for fname in os.listdir(target_dir)
                       if fname.endswith('.png') and not fname.startswith('.')])

plt.axis('off')
plt.imshow(load_img(input_img_paths[9]))
plt.savefig('./plots/image_segmentation_load_img.png')

target_array = img_to_array(load_img(target_paths[9], color_mode='grayscale'))
normalized_array = (target_array.astype('uint8') - 1) * 127
plt.axis('off')
plt.imshow(normalized_array[:, :, 0])
plt.savefig('./plots/image_segmentation_target_img.png')
