import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sample_size_per_class = 1000
negative_samples = np.random.multivariate_normal(mean=[0, 3],
                                                 cov=[[1, 0.5], [0.5, 1]],
                                                 size=sample_size_per_class)

positive_samples = np.random.multivariate_normal(mean=[3, 0],
                                                 cov=[[1, 0.5], [0.5, 1]],
                                                 size=sample_size_per_class)

inputs = np.vstack((negative_samples, positive_samples)).astype('float32')

targets = np.vstack((np.zeros((sample_size_per_class, 1), dtype='float32'),
                     np.ones((sample_size_per_class, 1), dtype='float32')))

# TODO plot is not working for remote intrepreters. Fix it
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()

# linear classifier in pure python
# our input is just 2 features created above
input_dim = 2

# this is a binary classifier, so our output dimension is just 1.
output_dim = 1

W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))


def forward_pass(input_features):
    """

    :param input_features:
    :return:
    """
    return tf.matmul(input_features, W) + b


def loss_function(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    per_sample_losses = tf.square(y_true - y_pred)
    return tf.reduce_mean(per_sample_losses)


learning_rate = 0.1


def training_step(input_features, y_true):
    """

    :param input_features:
    :param y_true:
    :return:
    """
    with tf.GradientTape() as tape:
        predictions = forward_pass(input_features)
        loss = loss_function(y_true, predictions)
    gradient_loss_wrt_W, gradient_loss_wrt_b = tape.gradient(loss, [W, b])
    W.assign_sub(learning_rate * gradient_loss_wrt_W)  # equal to W -= learning_rate * gradient_wrt_W
    b.assign_sub(learning_rate * gradient_loss_wrt_b)
    return loss


for step in range(40):
    loss = training_step(inputs, targets)
    print(f'Loss at step {step}: {loss:.4f}')

predictions = forward_pass(inputs)
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()

x = np.linspace(-1, 4, 100)
y = -(W[0]/W[1]) * x + (0.5 - b) / W[1]
plt.plot(x, y, '-r')
plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.show()
