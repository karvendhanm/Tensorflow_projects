# comparing/timing the speed of execution between numpy element-wise operations
# and our own implementation using a for loop

import numpy as np
import time

def naive_relu(x):

    # we need a rank 2 tensor
    assert len(x.shape) == 2

    # avoid overwriting the input tensor
    x = x.copy()

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):

            x[i, j] = max(x[i, j], 0.)
    return x


def naive_add(x, y):
    # we need a rank 2 tensor
    assert len(x.shape) == 2

    # we need both the inputs to be rank 2 tensor
    assert len(x.shape) == len(y.shape)

    # avoid overwriting the input tensor
    x = x.copy()

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]

    return x



x = np.random.random((20, 100))
y = np.random.random((20, 100))

t0 = time.time()
for _ in range(1000):
    z = x + y
    z = np.maximum(x, 0.)
print('took: {0:.3f} s'.format(time.time() - t0))

t0 = time.time()
for _ in range(1000):
    z = naive_add(x, y)
    z = naive_relu(x)
print('took: {0:.3f} s'.format(time.time() - t0))
