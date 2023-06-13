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