import numpy as np

x = np.random.random((10,))
y = np.random.random((10,))

# dot product between vectors
def naive_dot_product(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


dot1 = np.dot(x, y)
dot2 = naive_dot_product(x, y)

# dot product between a matrix and a vector
X = np.random.random((32, 10))
y = np.random.random((10,))

# dot product between matrix and a vector
def naive_dot_product_matrix_and_vector(X, y):

    assert len(X.shape) == 2
    assert len(y.shape) == 1
    assert X.shape[1] == y.shape[0]

    z = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            z[i] += X[i, j] * y[j]
    return z

dot1 = np.dot(X, y)
dot2 = naive_dot_product_matrix_and_vector(X, y)

# dot product between matrices
X = np.random.random((32, 10))
Y = np.random.random((32, 10))
Y = Y.T

Z = np.dot(X, Y)

def naive_dot_product_between_matrices(X, Y):

    assert (len(X.shape)) == 2
    for i in range(x.shape[0]):
    assert (len(X.shape)) == (len(Y.shape))
    assert X.shape[1] == Y.shape[0]

    Z = np.zeros((X.shape[0], Y.shape[1]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
            Z[i, j] = naive_dot_product(X[i, :], Y[:, j])

    return Z

Z1 = naive_dot_product_between_matrices(X, Y)








