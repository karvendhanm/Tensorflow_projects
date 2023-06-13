import numpy as np

# rank 1 tensor or scalar
x = np.array(12)
x
x.ndim
x.shape

# rank 1 tensor or vector
x = np.array([12, 3, 6, 14, 7])
x
x.ndim

# rank 2 tensor
x = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
x.ndim
x.shape

# rank 3 tensor
x = np.array([
    [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
    ],
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ],
    [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
    ],
    [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
    ],
])
x.shape
x.ndim

x.dtype


from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images.ndim
train_images.shape
train_images.dtype

# displaying the fifth image
digit = train_images[4]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# tensor slicing
arr_ = np.array([
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [1, 2, 3, 4]
])

arr_.shape
arr_.ndim

arr_[:, :]
arr_[2:, 2:]
arr_[1:-1, 1:-1]