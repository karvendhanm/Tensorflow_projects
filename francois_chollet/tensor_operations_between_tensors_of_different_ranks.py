# broadcasting to perform tensor operations between tensors of different ranks
import numpy as np

X = np.random.random((32, 10))
y = np.random.random((10,))

print(f'the rank of tensor X is: {X.ndim}')
print(f'the rank of tensor y is: {y.ndim}')

# tensor operations performed between a matrix and a vector
M = X + y

# mental model on how tensor operations between tensors of different ranks is
# accomplished by Numpy
Y = np.expand_dims(y, axis=0)
Y = np.concatenate([Y] * 32, axis=0)
N = X + Y

def naive_add_matrix_and_vector(X, y):

    assert len(X.shape) == 2
    assert len(y.shape) == 1
    assert X.shape[1] == y.shape[0]

    X = X.copy()
    for i in range(X.shape[0]):
        for j in range(X. shape[1]):

            X[i, j] += y[j]
    return X

P = naive_add_matrix_and_vector(X, y)

X = np.random.random((64, 3, 32, 10))
Y = np.random.random((32, 10))

Z = np.maximum(X, Y)

# mental model: how the above np.maximum was accomplished given
# both the tensors X and Y have different ranks.

Y_1 = np.expand_dims(Y, axis=0)
Y_2 = np.concatenate([Y_1]*3, axis=0)
Y_3 = np.expand_dims(Y_2, axis=0)
Y_4 = np.concatenate([Y_3] * 64, axis=0)
Z1 = np.maximum(X, Y_4)































